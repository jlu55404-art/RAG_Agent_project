import re  # todo re 是 Python 的正则表达式库，用于文本匹配和替换
import torch  # todo PyTorch 深度学习框架，用来运行 BERT 模型做推理
from typing import List  # todo List 是 Python 类型提示，表示列表类型
from transformers import AutoTokenizer, AutoModelForTokenClassification  # todo HuggingFace transformers 库：AutoTokenizer 分词器，AutoModelForTokenClassification 用于序列标注的模型
from langchain_text_splitters import CharacterTextSplitter  # todo LangChain 的字符级文本分割器基类

# 本地模型路径
model_path = '../models/nlp_bert_document-segmentation-chinese-base'  # todo 指向本地下载好的 BERT 段落分割模型（不是从网上下载，是本地加载）


class AliTextSplitter(CharacterTextSplitter):
    # todo 这个类的作用：用阿里达摩院训练的 BERT 模型来智能分割中文文章的段落
    # todo 原理：模型判断每个句子是不是"新段落的开头"，根据概率来切分
    # todo 继承自 LangChain 的 CharacterTextSplitter，可以融入 LangChain 生态
    def __init__(self, pdf: bool = False, device: str = "cpu", **kwargs):
        # todo 初始化函数
        # todo pdf=True 时会对 PDF 提取的文本做预处理（去多余换行和空格）
        # todo device="cpu" 用 CPU 推理，"cuda" 用 GPU 推理（速度快但需要显卡）
        super().__init__(**kwargs)  # todo 调用父类初始化
        self.pdf = pdf  # todo 是否是 PDF 模式（PDF 文本需要额外清洗）
        self.device = device  # todo 模型运行在什么设备上（cpu 或 cuda）

        # 加载模型和分词器（只在初始化时加载一次）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # todo 加载分词器：把中文句子拆成一个个 token（词或字）
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)  # todo 加载 BERT 模型：判断每个位置是不是段落边界

        # 将模型移动到指定设备
        self.model.to(self.device)  # todo 把模型搬到指定设备（CPU 或 GPU）
        self.model.eval()  # 设置为评估模式  # todo eval() 模式：关闭 dropout 等训练专用操作，只做推理

    def _split_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # todo 这个方法的作用：把一大段文本按中文标点拆成一句一句的
        # 使用标点符号作为分割标志
        sentence_delimiters = ['。', '！', '？', '；', '：', '…']  # todo 这些是中文句子结束的标点符号

        # 构建正则表达式模式
        pattern = '([' + ''.join(re.escape(d) for d in sentence_delimiters) + '])'  # todo 构建正则：匹配任意一个句子结束标点，括号保留标点

        # 分割文本
        parts = re.split(pattern, text)  # todo 用正则切文本，标点会单独保留

        # 重组句子
        sentences = []  # todo 存放拆分好的句子
        current_sentence = ""
        for i in range(0, len(parts), 2):  # todo 步长为2：偶数位置是文字，奇数位置是标点
            if i + 1 < len(parts):
                sentence = parts[i] + parts[i + 1]  # todo 文字 + 标点 = 完整句子
            else:
                sentence = parts[i]  # todo 最后一个片段可能没有标点（原文没写完）
            if sentence.strip():  # todo 如果句子不为空
                sentences.append(sentence.strip())  # todo 去掉首尾空格后加入句子列表

        # 处理没有标点的情况
        if not sentences and text.strip():  # todo 如果没有任何标点，但文本不为空
            sentences = [text.strip()]  # todo 整段当成一个句子

        return sentences  # todo 返回拆分好的句子列表

    def _segment_with_model(self, sentences: List[str], threshold: float = 0.5) -> List[str]:
        """使用BERT模型进行段落分割"""
        # todo 这个方法的作用：用 BERT 模型判断每个句子是不是新段落的开始，然后切段
        # todo threshold 是阈值（0到1之间），概率超过这个值就认为应该在这里切新段落
        if not sentences:
            return []  # todo 没句子就返回空列表

        # 如果只有一个句子，直接返回
        if len(sentences) == 1:
            return [sentences[0]]  # todo 只有一个句子，不需要分割

        # 获取每个句子是段落开头的概率
        boundary_probs = []  # todo 存储每个句子"是新段落开头"的概率

        # 逐句处理（避免内存溢出）
        with torch.no_grad():  # todo 关闭梯度计算，推理不需要梯度，省内存、速度快
            for i, sentence in enumerate(sentences):
                # 编码当前句子
                inputs = self.tokenizer(  # todo 把中文句子转成 BERT 能理解的数字（token ids）
                    sentence,
                    return_tensors='pt',  # todo 返回 PyTorch 张量格式
                    truncation=True,  # todo 句子太长就截断
                    max_length=512,  # todo 最多 512 个 token（BERT 的最大输入长度）
                    padding=True  # todo 不够长的自动补零
                )

                # 移动到相同设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}  # todo 把输入数据也搬到和模型一样的设备上

                # 模型推理
                outputs = self.model(**inputs)  # todo 把输入喂给模型，拿到输出（每个 token 的分类分数）
                probs = torch.softmax(outputs.logits, dim=-1)  # todo softmax 把分数转成概率（0到1之间）

                # 获取边界概率（假设label 1表示新段落开始）
                # 注意：需要根据实际模型输出的标签映射调整
                sentence_prob = probs[0, 0, 1].item()  # [batch, token, class]  # todo 取第一个 token 的第2类（label=1）的概率，代表"是新段落开头"的置信度
                boundary_probs.append(sentence_prob)  # todo 记录这个概率

        # 根据概率决定分割点
        paragraphs = []  # todo 存放最终切好的段落
        current_paragraph = []  # todo 当前正在攒的段落（还没到切分点）

        for i, sentence in enumerate(sentences):
            current_paragraph.append(sentence)  # todo 把当前句子加入当前段落

            # 判断是否需要在这里分割（下一个句子是段落开头）
            if i < len(sentences) - 1 and boundary_probs[i + 1] >= threshold:  # todo 下一个句子的边界概率超过阈值，说明应该在这里切段
                paragraphs.append(''.join(current_paragraph))  # todo 把攒好的句子拼成一个段落
                current_paragraph = []  # todo 清空，开始攒下一个段落

        # 添加最后一个段落
        if current_paragraph:  # todo 最后一段可能还没凑够，也要加进去
            paragraphs.append(''.join(current_paragraph))

        return paragraphs  # todo 返回切好的段落列表

    def split_text(self, text: str, threshold: float = 0.5) -> List[str]:
        """
        分割文本为段落

        Args:
            text: 输入文本
            threshold: 段落边界判断阈值（0-1之间，值越大越保守）

        Returns:
            分割后的段落列表
        """
        # todo 这是对外暴露的主方法：输入一段长文本，输出切好的段落列表
        # todo threshold 越大，切得越保守（段落越长）；越小，切得越细（段落越短）
        # PDF模式下的文本预处理
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)  # todo PDF 文本预处理：把3个及以上的换行合并成1个
            text = re.sub(r'\s', " ", text)  # todo 把所有空白字符替换为空格
            text = re.sub(r"\n\n", "", text)  # todo 去掉连续两个换行符

        # 先分割成句子
        sentences = self._split_sentences(text)  # todo 第一步：把文本拆成句子

        # 使用模型进行语义分割
        if len(sentences) > 1:
            paragraphs = self._segment_with_model(sentences, threshold)  # todo 第二步：用 BERT 模型判断每个句子间要不要切段
        else:
            # 如果句子太少，直接返回
            paragraphs = [text] if text.strip() else []  # todo 只有一句话，就不切了

        # 清理输出格式
        cleaned_paragraphs = []  # todo 存放清洗后的段落
        for para in paragraphs:
            # 移除多余的换行符和制表符
            cleaned = re.sub(r'\n+', '\n', para.strip())  # todo 把段落里多余的连续换行合并成一个
            if cleaned:
                cleaned_paragraphs.append(cleaned)  # todo 加入清洗后的段落

        return cleaned_paragraphs  # todo 返回最终切好并清洗过的段落列表


# 为了兼容原代码的简单调用方式，提供一个简化版本
class SimpleAliTextSplitter:
    """简化版本，不依赖langchain"""
    # todo 这个类是 AliTextSplitter 的简化包装版，不需要继承 LangChain，用法更简单
    # todo 如果你不想依赖 LangChain 框架，可以用这个简化版

    def __init__(self, pdf: bool = False, device: str = "cpu"):
        # todo 初始化函数，参数和 AliTextSplitter 一样
        self.pdf = pdf
        self.device = device

        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # todo 加载分词器
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)  # todo 加载 BERT 段落分割模型
        self.model.to(self.device)  # todo 模型搬到指定设备
        self.model.eval()  # todo 设为推理模式

        # 创建实际的分割器
        self.splitter = AliTextSplitter(pdf=pdf, device=device)  # todo 内部还是创建 AliTextSplitter 实例来做实际分割

    def split_text(self, text: str) -> List[str]:
        """简化的接口，直接调用分割器"""
        # todo 简化版接口：只需要传文本，不需要调 threshold 参数
        return self.splitter.split_text(text)  # todo 委托给 AliTextSplitter 做实际分割


if __name__ == '__main__':
    # 测试文本
    test_text = '''移动端语音唤醒模型，检测关键词为"小云小云"。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用"basetrain + finetune"的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景"小云小云"数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。'''

    # 创建分割器实例
    model_split = AliTextSplitter(device="cpu")  # todo 用 CPU 模式创建分割器实例
    # 可以改为 "cuda" 使用GPU（需要额外安装gpu版本的torch）

    # 执行分割
    result = model_split.split_text(text=test_text, threshold=0.5)  # todo 对测试文本执行段落分割，阈值 0.5

    # 打印结果
    print("原始文本长度:", len(test_text))
    print("分割后段落数:", len(result))
    print("\n分割结果:")
    for i, para in enumerate(result, 1):
        print(f"\n段落 {i}:")
        print(para)
        print(f"段落长度: {len(para)} 字符")
        print("-" * 50)