# todo 导入标准库：json 读评估数据文件，os 与操作系统交互
import json, os
# todo 1. 核心导包
# todo ragas 是一个专门评估 RAG（检索增强生成）系统好坏的 Python 库
# todo evaluate 是它的核心函数：给数据+指标，它自动算出各项评测分数
from ragas import evaluate# ragas.evaluate函数，用于执行RAG评估
# todo RunConfig：运行配置，控制评估时的并发数、超时时间等
from ragas.run_config import RunConfig # 配置大模型并发数、超时时间
# 导入ragas的评估指标，包括忠实度、答案相关性、上下文精确率、上下文召回率
# todo 下面4个是 RAG 系统最常用的评估指标
from ragas.metrics import (
    faithfulness, # 忠实度: 答案是否完全基于提供的上下文(判断幻觉)
    answer_relevancy, # 答案相关性: 答案是否紧扣问题(答的准不准)
    context_precision, # 上下文精确率: 上下文是否为有效内容、排序是否合理(噪音情况)
    context_recall # 上下文召回率: 上下文是否包含回答问题所需全部信息(信息完整度)
)
# todo Dataset：HuggingFace datasets 库的数据集类，ragas 要求数据必须是这个格式
from datasets import Dataset  # Dataset类构建RAGAS所需的数据格式

# todo 2.模型及相应配置:
#  选择用于评估的LLM(大语言模型)和嵌入模型(用于计算相似度, 用于语义计算)
''' ⚠️ 无论使用云上模型还是本地模型，都请耐心等待，都很慢 '''

# TODO 使用 Ollama 本地模型（qwen2.5:7b + mxbai-embed-large）
# todo ChatOllama：LangChain 封装的本地 Ollama 大模型接口，不需要联网调用 API
# todo OllamaEmbeddings：LangChain 封装的本地 Ollama 嵌入模型，把文本转成向量
from langchain_ollama import ChatOllama, OllamaEmbeddings
# todo 2.配置评估模型: 本地Ollama模型
# 大语言模型：qwen2.5:7b
# todo llm 对象用于 ragas 评估中被调用来判断答案质量（比如判断"这个答案是不是瞎编的"）
llm = ChatOllama(
    model="qwen2.5:7b",  # todo 使用 Ollama 本地运行的 qwen2.5 7B 模型
    base_url="http://localhost:11434",  # todo Ollama 服务的默认端口是 11434
    # 修改默认配置，可一定程度提升速度
    # temperature=0,          # todo 评估固定随机性，稳定+提速
    # num_ctx=1024,           # 强制上下文窗口1024，可以降低显存
    # num_predict=512,        # 如果评估不需要长文本，就限制最大输出
    # request_timeout=600,    # 接口超时 10分钟，防止中途断开
    # extra_body = {
    #     "format": "json" # todo 强制模型按json规范输出
    # }
)
# 向量嵌入模型：mxbai-embed-large:latest
# todo embeddings 用于把文本转成向量然后计算相似度（比如判断两段话语义上有多像）
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large:latest",
    base_url="http://localhost:11434"
)
# 配置并发数、超时时间
# todo RunConfig：控制 ragas 评估时的性能参数
run_config = RunConfig(
    # timeout=600,      # 单次请求超时，单位秒
    max_workers=1,      # 本地ollama强制为串行、不并发！（本地模型显存有限，同时跑多个容易崩）
    max_retries=2       # 失败重试次数
)

# # TODO 使用 阿里云百炼 模型（qwen3.7-max + text-embedding-v4）
# from langchain_community.chat_models import ChatTongyi
# from langchain_community.embeddings import DashScopeEmbeddings
# llm = ChatTongyi(
#     model="qwen3.7-max",
#     api_key=os.getenv('AliApiKey')
# )
# embeddings = DashScopeEmbeddings(
#     model="text-embedding-v4",
#     dashscope_api_key=os.getenv('AliApiKey'),
# )
# # 配置并发数、超时时间
# run_config = RunConfig(
#     # timeout=120,      # 单次请求超时，单位秒
#     max_workers=4,      # 并发请求数，根据API限流调整（一般 4~16）
#     max_retries=2       # 失败重试次数
# )

# todo 3.加载评估数据集:
# 读取包含问题, 答案, 上下文, 真实答案的JSON文件
# todo 这个 JSON 文件里每条数据包含：question（问题）、answer（系统生成的答案）、
# todo context（检索到的上下文文档列表）、ground_truth（人工标注的标准答案）
with open('rag_evaluate_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    # dict or list = json.loads(json_str)
print(f'加载的评估样本数量: {len(data)}')
if data:
    print(f'查看第一条评估样本: {data[0]}')

# todo 4.转换数据格式:
# 将原始数据转换为RAGAS所需的DataSet格式
# todo ragas 库要求输入的数据必须是用 datasets.Dataset 格式，不能直接用 Python 列表
eval_data = {
    'question': [item['question'] for item in data],         # 问题列表
    'answer': [item['answer'] for item in data],              # 生成答案列表（RAG系统实际回答的）
    'contexts': [item['context'] for item in data],          # 上下文列表（检索到的文档内容）
    'ground_truth': [item['ground_truth'] for item in data]   # 标准答案列表（人工写的正确答案）
}
# todo Dataset.from_dict() 把 Python 字典转成 HuggingFace 的 Dataset 对象
dataset = Dataset.from_dict(eval_data)
print(f'转换后的数据集: {dataset}')

# todo 5. 执行评估:
# todo evaluate() 是 ragas 的核心函数：它会用大模型逐个判断每条数据的4个指标，然后汇总成平均分
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,       # 忠实度：答案有没有瞎编（幻觉检测）
        answer_relevancy,    # 答案相关性：答案有没有跑题
        context_precision,   # 上下文精确率：检索到的文档里有多少是真正有用的
        context_recall       # 上下文召回率：该检索到的文档是不是都检索到了
    ],
    llm=llm,                # todo 用这个大模型来判断质量（模型自己评估自己可能有点矛盾，但这是业界做法）
    embeddings=embeddings,  # todo 用这个嵌入模型计算语义相似度
    run_config=run_config   # 加入配置
)

# todo 6. 输出并保存评估结果
print(f'RAGAS评估结果: {result}')
# 结果保存为CSV文件
# todo result.to_pandas() 把评估结果转成 Pandas DataFrame，再 to_csv 保存到 CSV 文件
# todo encoding='utf-8-sig' 带 BOM 的 UTF-8，确保 Excel 打开不乱码
result.to_pandas().to_csv('rag_as.csv', index=False, encoding='utf-8-sig')
print("✅ 评估结果已保存至 rag_as.csv")