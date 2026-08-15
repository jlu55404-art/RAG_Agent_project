""" 意图识别
┌─────────────────────────────────────────────────────────────────┐
│                    QueryClassifier 函数调用关系                    │
└─────────────────────────────────────────────────────────────────┘

═══════════════ 初始化阶段 ═══════════════

  __init__()
     │
     └──► _load_model()
              │
              ├─ 有微调模型? ──► 加载 bert-query-classifier-500
              └─ 没有?     ──► 加载 bert-base-chinese 预训练模型


═══════════════ 训练阶段（_train_model 是总指挥）═══════════════

  _train_model()
     │
     ├─① _get_train_data_split()     读JSON → 80/20划分
     │       输出: train_texts, val_texts, train_labels, val_labels
     │
     ├─② _preprocess_data() ×2       文本→tensor, 标签→[0,1]
     │       输出: encodings, labels
     │
     ├─③ _create_dataset() ×2        编码+标签 → Dataset对象
     │       输出: train_dataset, val_dataset
     │
     ├─④ _get_training_args()        构造训练超参数对象
     │       输出: training_args
     │
     ├─⑤ _get_trainer()              组装Trainer(模型+数据+参数)
     │       输出: trainer
     │
     ├─⑥ trainer.train()             开练！
     │
     ├─⑦ model.save_pretrained()     保存模型+分词器
     │
     └─⑧ _evaluate_model()           用验证集评估
              │
              ├─ tokenizer()          编码验证文本
              ├─ _create_dataset()    构造Dataset
              ├─ Trainer.predict()    预测
              └─ classification_report() + confusion_matrix()


═══════════════ 推理阶段（项目实际调用的）═══════════════

  predict_category(query)       ← rag_system.py 调用这个
     │
     ├─ tokenizer(query)        文本→tensor
     ├─ model(**encoding)       前向推理
     ├─ torch.argmax()          取最高分类别
     └─ return "通用知识"/"专业咨询"


═══════════════ 一句话总结 ═══════════════

  训练时：_train_model 按 ①→⑧ 顺序串起所有子函数
  推理时：只用 predict_category，跟训练流程完全独立
"""



import os, json  # todo os: 用于检查文件/目录是否存在; json: 用于读写JSON格式的标注数据
import torch  # todo 大白话: PyTorch深度学习框架的核心库, 一切张量运算、模型推理都靠它
import numpy as np  # todo 大白话: 科学计算库, 这里主要用它做数组操作（如argmax、mean等）
from transformers import BertTokenizer, BertForSequenceClassification
# todo BertTokenizer: 大白话——"分词器", 把一句中文拆成一个个词/字, 然后转成BERT能理解的数字ID
# todo BertForSequenceClassification: 大白话——"用于文本分类的BERT模型", 在BERT基础上加了一个分类头, 输出两个分数（通用知识/专业咨询）
# TrainingArguments: 负责定义训练的各种参数和配置
#   比如: 训练多久, 用多大批次, 保存到哪等
#   相当于: 训练任务的说明书. 它是Trainer的"操作指南"
# Trainer: 负责实际执行训练任务
#   用模型, 数据, TrainingArguments的配置 跑训练
from transformers import Trainer, TrainingArguments
# todo Trainer: 大白话——"训练教练", 你把模型、数据、训练参数塞给它, 它帮你自动跑完整个训练循环（前向传播→算loss→反向传播→更新参数）
# todo TrainingArguments: 大白话——"训练配置单", 里面写着练几轮、每批喂多少数据、学习率多少、啥时候保存等
from sklearn.model_selection import train_test_split # 划分训练集、验证集
# todo train_test_split: 大白话——"切分函数", 把一整份数据按比例切成两半, 比如80%拿去训练, 20%留着验证模型好不好
from sklearn.metrics import classification_report, confusion_matrix
# todo classification_report: 大白话——"分类成绩单", 输出每类的精确率、召回率、F1分数等指标
# todo confusion_matrix: 大白话——"混淆矩阵", 一张表格告诉你"预测对了多少、把A错判成B的有多少"

from base import logger, Config
# todo logger: 大白话——"日志记录器", 把程序运行时的关键信息打印出来（比如模型加载成功、训练完成等）
# todo Config: 大白话——"全局配置类", 统一管理项目中所有路径和参数, 比如模型放哪、训练数据在哪、用CPU还是GPU

# TODO 定义QueryClassfier类:
#  封装BERT查询分类的完整流程
#  1. 加载预训练、微调模型
#  2. 处理数据
#     - 获取训练集、测试集
#     - 预处理 文本转张量，标签转[1,0]
#     - 组装Dataset数据集
#  3. 构建训练参数
#  4. 构建训练对象
#  5. 微调训练
#  6. 保存模型
#  7. 评估模型
#  8. 使用模型进行预测，实现检索的意图识别：
#     → 能够把输入进行分类：{"通用知识": 0, "专业咨询": 1}

# todo 这个类是"查询意图分类器"的核心大脑。
# todo 大白话：它的工作就是看用户输入的一句话（比如"什么是机器学习"），
# todo 然后判断这句话属于"通用知识"还是"专业咨询"。
# todo 它封装了BERT模型从加载→训练→保存→推理的完整生命周期。
class QueryClassifier:

    # todo 这个方法的作用是：初始化分类器, 加载分词器、配置设备、建立标签映射, 最后调用_load_model()把模型加载进来
    def __init__(self,
                 new_model_path=Config.BERT_QUERY_PATH):
        """初始化方法:
            配置模型路径, 加载分词器, 选择设备, 定义标签映射
        :param new_model_path: 模型保存的路径.
        """
        # 配置训练后模型路径
        self.new_model_path = new_model_path
        # 分词器
        # todo BertTokenizer.from_pretrained(): 大白话——从指定路径加载预训练好的"分词词汇表"。
        # todo 这个词汇表里记录了每个汉字/词对应哪个数字ID, 比如"我"→2769, "学"→2110。
        # todo BERT只认识数字, 所以任何中文都得先通过分词器变成一串数字ID。
        self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_PATH)
        # 预训练模型对象 在_load_model()方法中定义
        self.model = None  # todo 这个变量保存真正的BERT分类模型, 初始为None, 后续由_load_model()赋值
        # 训练器对象 在_get_trainer方法中定义
        self.trainer = None  # todo 这个变量保存Trainer对象, 只在训练阶段使用, 推理阶段用不到
        # 确定设备
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = Config.DEVICE
        # todo torch.device: 大白话——指定模型和张量在哪个硬件上跑。
        # todo "cuda"=NVIDIA显卡(GPU),"cpu"=中央处理器。GPU做矩阵运算比CPU快几十到上百倍。
        # 定义标签的映射关系 文本→数字
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        # todo label_map: 大白话——"标签对照表"。BERT模型输出的是数字(0或1)，但人要看中文。
        # todo 所以需要这个字典做翻译：训练时把中文→数字喂给模型, 推理时把数字→中文返回给用户。
        # 加载模型
        self._load_model()


    # todo 这个方法的作用是：加载BERT模型。优先用我们微调过的（有经验、更聪明），
    # todo 如果微调模型不存在，就用原始的预训练模型（有基础知识但没专门学过分类任务）。
    def _load_model(self):
        """加载模型：从指定路径加载 微调模型, 若不存在则加载 预训练模型"""
        # 判断微调模型是否有效：目录存在 且 包含模型权重文件
        # todo model.safetensors / pytorch_model.bin: 大白话——这是BERT模型的"大脑文件"，
        # todo 里面存着几亿个参数的具体数值（权重），没有这个文件模型就是个空壳。
        has_finetuned_model = (
            os.path.exists(self.new_model_path) and
            (os.path.exists(os.path.join(self.new_model_path, 'model.safetensors')) or
             os.path.exists(os.path.join(self.new_model_path, 'pytorch_model.bin')))
        )

        if has_finetuned_model:
            # 加载已微调的模型
            # todo BertForSequenceClassification.from_pretrained(): 大白话——从指定路径加载一个"带分类头的BERT模型"。
            # todo 这个分类头是一个全连接层，输入768维（BERT的输出维度），输出2维（两个类别）。
            # todo 微调过的模型已经在我们的数据上训练过了，直接就能用。
            self.model = BertForSequenceClassification.from_pretrained(
                self.new_model_path)
            self.model.to(self.device) # 模型指定使用的设备 gpu or cpu
            # todo .to(device): 大白话——把模型"搬运"到指定硬件上。如果在GPU上就搬到显存里，在CPU上就搬到内存里。
            logger.info(f"已加载微调模型：{self.new_model_path}")
        else:
            # 加载预训练的模型
            # todo num_labels=len(self.label_map): 大白话——告诉BERT"我要分几类"。
            # todo 这里len(self.label_map)=2，所以分类头输出2个分数：[通用知识得分, 专业咨询得分]。
            self.model = BertForSequenceClassification.from_pretrained(
                Config.BERT_PATH, # 基座模型的路径
                num_labels=len(self.label_map), # 标签的数量
            )
            self.model.to(self.device) # 模型指定使用的设备 gpu or cpu
            logger.info(f'已加载预训练模型：{Config.BERT_PATH}')

    # todo 这个方法的作用是：测试初始化是否成功，打印设备、路径、模型结构等信息，方便调试。
    def _test_INIT(self):
        print('测试 ➡️ 初始化功能：')
        print('使用训练的设备是：', self.device)
        print('预训练模型路径:', Config.BERT_PATH)
        print('新模型保存路径:', self.new_model_path)
        print('当前模型状态：', self.model)
        # 在输出的最后能看到 in_features=768 维, out_features=2 分两类

    # todo 1. --------数据预处理1. 获取 训练集、测试集--------
    # todo 这个方法的作用是：从JSON文件读取标注数据，拆成训练集（80%）和验证集（20%），返回四样东西。
    def _get_train_data_split(
            self,
            data_file=Config.BERT_TRAIN_DATA):
        """获取训练集、测试集"""
        # 判断数据集文件是否存在，不存在直接报错
        if not os.path.exists(data_file):
            logger.error(f'数据集文件不存在：{data_file}')
            raise FileNotFoundError(f'数据集文件不存在：{data_file}')

        # 加载数据集：从json文件中获取查询文本、标签 → [dict]
        # todo json.loads(): 大白话——把JSON格式的字符串（'{"query":"你好","label":"通用知识"}'）
        # todo 转成Python字典（{"query":"你好","label":"通用知识"}），这样代码里就能用dict["query"]取东西了。
        with open(data_file, 'r', encoding='utf8') as f:
            # 遍历每一行json，json.loads()转成dict，放到列表中
            data = [json.loads(value) for value in f.readlines()]

        # 提取文本和标签 [dict]-> [query_str], [label_str]
        # todo 列表推导式: 大白话——遍历data列表里的每个dict，提取出query字段组成一个纯文本列表
        texts = [item['query'] for item in data]
        labels = [item['label'] for item in data]

        # 划分训练集、验证集
        # random_state=42 随机种子，保证每次划分结果一致
        # test_size=0.2 验证集占比20%
        # 训练集文本、验证集文本、训练集标签、验证集标签
        # todo train_test_split: 大白话——把数据随机洗牌后按8:2比例切开。
        # todo 训练集用来"学"（让模型拟合），验证集用来"考"（看模型学得怎么样）。
        # todo random_state=42: 大白话——"随机种子"，相当于给洗牌过程一个固定编号，
        # todo 下次还用42这个编号，洗出来的结果一模一样，保证实验可复现。
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42)

        return train_texts, val_texts, train_labels, val_labels

    def _test_GET_TRAIN_DATA_SPLIT(self):
        print('测试 ➡️ 获取训练集、测试集：')
        train_texts, val_texts, train_labels, val_labels = (
            self._get_train_data_split()
        )
        print('训练集文本数量：', len(train_texts))
        print('训练集标签数量：', len(train_labels))
        print('验证集文本数量：', len(val_texts))
        print('验证集标签数量：', len(val_labels))
        print('训练集文本前5行：', train_texts[:5])
        print('训练集标签前5行：', train_labels[:5])
        print('验证集文本前5行：', val_texts[:5])
        print('验证集标签前5行：', val_labels[:5])


    # todo 2. --------数据预处理2. 返回 文本编码和标签--------
    # todo 这个方法的作用是：把中文文本变成BERT能吃的"数字饲料"（张量），同时把中文标签变成0或1的数字。
    def _preprocess_data(self, texts, labels):
        """数据预处理：文本列表[str]→tensor; 标签字典→[0, 1]
        1. BERT-BASE-CHINESE构成的分词器对文本进行编码（分词, 截断, 填充）
        2. 将标签转换为数字
        :param texts: 待处理的文本列表 [str]
        :param labels: 文本对应的标签列表(通用知识 或 专业咨询) [str]
        :return:
            encodings: texts[str]→pytorch的张量
            labels: 标签字典→[0, 1]
        """
        # 文本编码：用bert-base-chinese构成的分词器
        # todo self.tokenizer(...): 大白话——分词器干活的核心步骤：
        # todo ①把句子切碎成字词（比如"什么是AI"→["什","么","是","AI"]）；
        # todo ②每个字词查词汇表找到对应数字ID（如"什"→752）；
        # todo ③在前面加[CLS]标记(101)，后面加[SEP]标记(102)——BERT规定必须这样；
        # todo ④所有句子统一长度：长的砍掉(max_length=128)，短的补0(padding='max_length')。
        encodings = self.tokenizer(
            texts, # 待处理的文本
            max_length=128, # 文本最大长度
            # todo max_length=128: 大白话——每条文本最多保留128个token（字/词），
            # todo BERT-base最多支持512，但128对短查询已经够了，而且算得更快。
            truncation=True, # 超过max_length长度就截断
            padding='max_length', # 不够max_length长度就填充
            # todo padding='max_length': 大白话——所有句子都补齐到128个位置，短的在后面填0。
            # todo 这是因为GPU处理数据要求所有输入必须一样长（像切好的薯条，不能有长有短）。
            return_tensors='pt' # 返回pytorch的张量
            # todo return_tensors='pt': 大白话——返回PyTorch张量格式（tensor），而不是Python列表。
            # todo 张量是PyTorch的基本数据类型，可以直接放到GPU上加速计算。
            # 如果不加return_tensors="pt"，返回list
            # pytorch张量类型可以直接进行训练
        )

        # 标签处理：文本标签转数字 [通用知识, 通用知识, 专业咨询, ..] → [0, 0, 1, ..]
        # self.label_map = {"通用知识": 0, "专业咨询": 1}
        labels = [self.label_map[label] for label in labels]

        # 返回编码和标签
        return encodings, labels

    def _test_PREPROCESS_DATA(self):
        print('测试 ➡️ 数据预处理：')
        texts = ["1024乘以768等于多少？", "专业咨询如何使用BERT进行分类？"]
        labels = ["通用知识", "专业咨询"]
        encodings, labels = self._preprocess_data(texts, labels)
        print('文本编码：', encodings) # dict
        print('标签：', labels) # 标签： [0, 1]


    # todo 3. --------数据预处理3. 返回 Dataset类型的数据集--------
    # todo 这个方法的作用是：把编码好的数据打包成PyTorch标准的Dataset对象，这样Trainer才能按批次取出数据喂给模型。
    def _create_dataset(self, encodings, labels):

        # 定义DataSet类 继承Pytorch的DataSet
        # todo torch.utils.data.Dataset: 大白话——PyTorch官方规定的"数据集模具"。
        # todo 继承它只需要实现三个方法：__init__(初始化)、__len__(告诉别人有多少条数据)、
        # todo __getitem__(取出第idx条数据)。Trainer训练时就是通过反复调用__getitem__来一条条取数据的。
        class DataSet(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                super().__init__() # 初始化
                self.encodings = encodings
                self.labels = labels

            def __len__(self): # 返回数据集长度的方法
                return len(self.labels)

            # 获取数据集的某个样本
            def __getitem__(self, idx):
                '''提取第idx条数据，添加张量形式的标签，返回字典'''
                item = {
                    # todo input_ids: 大白话——文本的数字ID序列，比如"你好"→[101, 872, 1962, 102]
                    'input_ids': self.encodings['input_ids'][idx],
                    # todo attention_mask: 大白话——"注意力掩码"，告诉你哪些位置有字(1)，哪些是填充的(0)。
                    # todo BERT计算时会忽略掉值为0的位置，只看真正有字的地方。
                    'attention_mask': self.encodings['attention_mask'][idx],
                    # todo token_type_ids: 大白话——"句子编号标记"，用来区分是第1句还是第2句。
                    # todo 这里只有单句分类，所以全是0。做句子对任务（如问答匹配）时才会用到0和1。
                    'token_type_ids': self.encodings['token_type_ids'][idx],
                    # todo torch.tensor(self.labels[idx]): 大白话——把数字标签(0或1)包装成PyTorch张量，
                    # todo 因为模型训练时loss函数要求标签必须是tensor格式。
                    'labels': torch.tensor(self.labels[idx])
                }
                return item

        # 返回DataSet类型的数据集
        return DataSet(encodings, labels)

    def _test_CREATE_DATASET(self):
        texts = ["1024乘以768等于多少？", "专业咨询如何使用BERT进行分类？"]
        labels = ["通用知识", "专业咨询"]
        encodings, labels = self._preprocess_data(texts, labels) #文本编码和标签
        dataset = self._create_dataset(encodings, labels)
        print('数据集长度：', len(dataset))
        print('数据集第一个样本：', dataset[0])
        # dataset[0] 调用的 DataSet.__getitem__ 函数
        '''{
            'input_ids': tensor([101, 2345, ..., 102]), # 只有第0条的输入tokensID
            'attention_mask': tensor([1, 1, ..., 1]),  # 只有第0条的注意力掩码
            'token_type_ids': tensor([0, 0, ..., 0]), # 只有第0条的句子边界标记
            'labels': tensor(0) # 只有第0条的标签
        }'''

    # todo 4. --------定义并返回训练参数对象--------
    # todo 这个方法的作用是：配置训练时的各种"开关和旋钮"——练多久、学多快、啥时候存盘等，返回一个配置对象。
    def _get_training_args(self):
        # 配置训练参数，返回训练参数对象
        # todo TrainingArguments: 大白话——"训练说明书"，里面每一项都是告诉Trainer"怎么练"。
        # todo 比如num_train_epochs=3就是"把全部数据反复看3遍"。
        training_args = TrainingArguments(
            output_dir=self.new_model_path,  # todo 检查点、日志、最终模型的保存路径
            # 如果目录不存在，Trainer 会自动创建
            # 训练中断后，可以通过指向该目录进行断点续训
            num_train_epochs=3,  # todo 训练轮数 2-4即可 过大容易导致过拟合
            # todo 大白话——epoch就是"把全部训练数据从头到尾看过一遍"。
            # todo 3轮=看3遍。太少学不够，太多就死记硬背了（过拟合）。
            per_device_train_batch_size=8,  # todo 每次训练迭代时，送入GPU/CPU的样本数量
            # todo 大白话——batch_size就是"一口吃几条数据"。8就是一次喂8条，
            # todo GPU并行处理这8条，算完再喂下一批8条。太小训练慢，太大显存爆。
            # 如果多显卡并行 每个设备的样本数量
            # 比如4张卡 全局Batch Size就是 8*4=32
            # 显存不足时请调小此值
            per_device_eval_batch_size=16,  # todo 验证(每个设备的)批次大小
            # 验证阶段不需要计算梯度，显存占用较小
            # 比训练时的batch size更大以加快评估速度
            warmup_steps=20,  # todo 学习率预热步数
            # 训练前20步 学习率从0线性增加到默认的初始学习率。
            # 防止初期破坏预权重分布，避免初始震荡
            # todo 大白话——"热身步数"。就像跑步前先慢走热身，模型前20步学习率从0慢慢涨起来。
            # todo 这样不会一上来就把预训练好的权重冲乱。
            # learning_rate=2e-5                # todo 默认学习率5e-5，即 0.00005
            # 通常在 1e-5 到 5e-5 之间调整
            # 过高会破坏预训练权重，过低则收敛太慢
            # todo 大白话——学习率就是"每次调整参数的步长"。太大容易跳过最优解，太小学得太慢。
            weight_decay=0.01,  # todo 权重衰减系数（L2 正则化系数）
            # 限制模型参数变得过大，防止过拟合 通常都是0.01
            # todo 大白话——"减肥惩罚"。如果模型参数变得太大，就额外加一个惩罚项，逼它保持小参数，防止过拟合。
            logging_steps=10,  # todo 每隔多少步保存一次日志
            eval_strategy="epoch",  # todo 评估频率 每轮训练结束后进行验证评估.
            save_strategy="epoch",  # todo 保存频率 每轮训练结束后, 保存模型检查点.
            # "no" 不进行评估
            # "steps" 每隔n步评估一次(需要参数eval_steps=n)
            load_best_model_at_end=True,  # todo 训练结束后加载验证集表现最好的模型.
            # 可以防止因为训练过度，保存频率>=评估频率才生效
            # eval_strategy必须设置为epoch或steps
            save_total_limit=1,  # todo 只保存一个检查点,即最优的模型,节省空间
            metric_for_best_model="eval_loss",  # todo 以验证集的损失作为最优模型的评判标准
            # 默认情况下Loss越小越好
            # 如果用准确率Accuracy作为指标 需要额外设置
            #   greater_is_better=True
            fp16=(True if self.device == 'cuda' else False),  # todo 是cuda就启用半精度训练
            # 显存占用减半 速度提升约2倍 通常不会损失精度
            # todo 大白话——fp16就是"用半精度浮点数(16位)代替全精度(32位)做计算"。
            # todo 好处：省一半显存、快两倍；代价：精度略微下降但通常可忽略。只有GPU(CUDA)才能用。
        )
        return training_args

    def _test_GET_TRAINING_ARGS(self):
        print('测试 ➡️ 查看完整的模型训练参数')
        training_args = self._get_training_args()
        print(training_args)


    # todo 5. --------定义并返回训练对象--------
    # todo 这个方法的作用是：组装一个"训练教练"（Trainer），把模型、数据、参数、评估函数全塞进去，返回这个教练。
    def _get_trainer(self, train_dataset, val_dataset, training_args):
        """构建并返回配置好的Trainer对象
        :train_dataset (torch.utils.data.Dataset): 用于模型微调的训练集
        :val_dataset (torch.utils.data.Dataset): 用于验证和监控模型过拟合情况的验证集
        :training_args :
        return: transformers.Trainer 模型训练器对象
        """
        # 自定义计算评估函数 返回准确率
        # todo compute_metrics: 大白话——"打分函数"。训练每轮结束后，Trainer会自动调用这个函数来给当前模型打分。
        # todo 它拿模型的预测结果和真实标签做对比，算出猜对了多少。
        def compute_metrics(eval_pred):
            """计算评估指标（计算准确率）"""
            # 接收模型输出的预测分数 和 真实标签
            logits, labels = eval_pred
            # todo logits: 大白话——模型输出的"原始分数"，还不是最终答案。
            # todo 比如对某条文本输出[2.3, -0.1]，意思是"通用知识得分2.3，专业咨询得分-0.1"。
            # 从分数里选出最高分的类别标签当做预测结果
            # axis=-1 表示最后一个维度，即类别维度
            # todo np.argmax(logits, axis=-1): 大白话——"找出每行中最大值的下标"。
            # todo [2.3, -0.1] → 下标0（因为2.3 > -0.1），所以预测为第0类=通用知识。
            predictions = np.argmax(logits, axis=-1)
            # 对比预测结果和真实标签，计算准确率
            accuracy = (predictions == labels).mean()
            # todo .mean(): 大白话——"算平均值"。如果100条里猜对了85条，accuracy=0.85=85%。
            print(f"accuracy: {accuracy:.4f}")
            '''举例：
            预测分数 logits = [
                [1.2, -0.5],   # 第1条样本：类别0分1.2，类别1分-0.5
                [0.3, 2.1],    # 第2条样本：类别0分0.3，类别1分2.1
                [-0.8, 0.6]    # 第3条样本：类别0分-0.8，类别1分0.6
            ]
            predictions = [0, 1, 1]
            真实标签 labels = [0, 1, 0]
            np.argmax(logits, axis=-1) # 在每行（最后一维）取最大值下标
            predictions == labels # 逐一比对，返回[True, True, False] → [1, 1, 0]
            .mean() 求平均 (1+1+0)÷3≈0.667 # 66.7%
            最终返回 {"accuracy": 0.6667}
            '''
            return {"accuracy": accuracy}

        # 构建并返回Trainer对象
        # todo Trainer: 大白话——HuggingFace提供的"自动化训练器"。
        # todo 你把模型、数据、配置塞给它，它帮你自动完成：前向传播→计算loss→反向传播→更新参数→评估→保存，
        # todo 这些本来需要手写几百行的训练循环，Trainer一行代码搞定。
        trainer = Trainer(
            model=self.model,  # 微调的模型 要被训练的基础模型
            args=training_args,  # 训练参数
            train_dataset=train_dataset,  # 训练集
            eval_dataset=val_dataset,  # 验证集
            compute_metrics=compute_metrics  # 评估函数
        )

        return trainer # 返回训练器对象


    def _test_GET_TRAINER(self):
        print('测试 ➡️ 创建模型训练对象')
        train_texts = ["1024乘以768等于多少？", "专业咨询如何使用BERT进行分类？"]
        train_labels = ["通用知识", "专业咨询"]
        val_texts = ["1024乘以768等于多少？", "专业咨询如何使用BERT进行分类？"]
        val_labels = ["通用知识", "专业咨询"]
        # 文本编码和标签
        train_encodings, train_labels = self._preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self._preprocess_data(val_texts, val_labels)
        # 获取训练集和验证集对象
        train_dataset = self._create_dataset(train_encodings, train_labels)
        val_dataset = self._create_dataset(val_encodings, val_labels)
        # 获取训练参数对象
        training_args = self._get_training_args()
        # 获取训练器对象
        trainer = self._get_trainer(train_dataset, val_dataset, training_args)
        print(trainer)
        '''第一次运行测试函数会自动创建存放新微调模型的空目录，
        再次运行就会报错，因为文件夹里是空的——可以删除该空目录
        '''


    # todo 6. --------模型训练、评估--------
    # todo 这个方法的作用是：总指挥！按顺序调用所有子函数，完成"读数据→预处理→打包→训练→保存→评估"的完整训练流水线。
    def _train_model(self, data_file=Config.BERT_TRAIN_DATA):
        """训练模型、评估模型"""
        # todo 1. 数据集：读JSON → 分成80%训练+20%验证
        train_texts, val_texts, train_labels, val_labels = self._get_train_data_split(data_file)
        # todo 2. 文本转编码 标签转数字：中文 → BERT能吃的数字tensor
        train_encodings, train_labels = self._preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self._preprocess_data(val_texts, val_labels)
        # todo 3. 构建Dataset类型的训练集和测试集：打包成PyTorch标准格式
        train_dataset = self._create_dataset(train_encodings, train_labels)
        val_dataset = self._create_dataset(val_encodings, val_labels)
        logger.info(f'数据集加载完成，请稍候 ...')
        # todo 4. 创建训练参数对象：设置练几轮、学多快等
        training_args = self._get_training_args()
        # todo 5. 创建训练器对象：把模型+数据+参数+评估函数组装起来
        trainer = self._get_trainer(train_dataset, val_dataset, training_args)
        # todo 6. 模型训练：这一行就是真正的"开练"！Trainer会自动完成前向传播→算loss→反向传播→更新参数
        trainer.train()
        # todo 7. 保存模型：把训练好的模型权重和分词器词汇表存到磁盘
        self.model.save_pretrained(self.new_model_path)
        self.tokenizer.save_pretrained(self.new_model_path)
        logger.info(f'模型保存完成')
        # todo 8. 用验证集评估模型的准确率：考一下模型，看它学得怎么样
        self._evaluate_model(val_texts, val_labels)


    # todo 这个方法的作用是：用验证集"考"一下模型，输出成绩单（准确率、混淆矩阵等）。
    # todo 跟训练无关，只是评测，所以不需要算梯度。
    def _evaluate_model(self, val_texts, val_labels):
        """用验证集评估模型性能
        :param val_texts: 待评估的文本列表(需要预测的输入文本)
        :param val_labels: 文本对应的真实标签列表(已转为数字形式), 0:通用知识, 1:专业咨询
        :return: 无,评估结果通过日志输出.
        """
        # 1. 文本进行分词编码
        encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt' # 返回张量
        )
        logger.info('开始评估模型性能')

        # 2. 将编码后的文本 和 真实标签封装为PyTorch的 DataSet对象.
        dataset = self._create_dataset(encodings, val_labels)
        # 3. 打印数据集信息: 验证数据集长度和第一个样本格式是否正确.
        logger.info(f'输出数据集样本数量: {len(dataset)}')
        logger.info(f'输出第一个样本的格式(编码 + 标签): {dataset[0]}')

        # 4. 训练器对象 仅传入模型
        # todo Trainer(model=self.model): 大白话——这里只传模型不传训练参数，表示"只做预测，不训练"。
        # todo 这样Trainer就不会尝试反向传播更新参数，只是纯推理模式。
        trainer = Trainer(model=self.model)

        # 5. 执行预测
        # todo trainer.predict(dataset): 大白话——把验证集数据喂给模型跑一遍前向传播，
        # todo 得到每个样本的预测分数(logits)。不更新模型参数。
        predictions = trainer.predict(dataset)

        # 6. 提示预测的标签
        # todo np.argmax(predictions.predictions, axis=-1): 大白话——从预测分数中取最高分的下标作为最终分类结果。
        # todo 比如某条样本输出[1.5, -0.3] → 取argmax得0 → 预测为"通用知识"。
        pred_labels = np.argmax(predictions.predictions, axis=-1)

        # 7. 输出分类结果报告
        # todo classification_report: 大白话——打印一张"成绩单"，包含每类的精确率(Precision)、
        # todo 召回率(Recall)、F1分数。精确率高=不乱猜，召回率高=不漏掉。
        report = classification_report(
            val_labels, # 真实标签
            pred_labels, # 预测标签
            # 把数字转成中文名称, 方便阅读.
            target_names=['通用知识', '专业咨询']
        )
        logger.info(f'输出分类结果报告: {report}')

        # 8. 输出混淆矩阵: 显示模型对每个类别的预测能力.
        # todo confusion_matrix: 大白话——打印一张2×2的表格：
        # todo  [ 真·通用知识被预测为通用知识的数量,     真·通用知识被预测为专业咨询的数量 ]
        # todo  [ 真·专业咨询被预测为通用知识的数量,       真·专业咨询被预测为专业咨询的数量 ]
        # todo 对角线上的数字越大越好（预测正确的多），非对角线越小越好（混淆的少）。
        logger.info("输出混淆矩阵:")
        # sklearn.metrics.confusion_matrix
        logger.info(confusion_matrix(val_labels, pred_labels))

    # TODO 专门的评估模型的测试函数 和 self._evaluate_model函数效果一致
    # TODO 这个函数大概看看就行了 其实就是调用模型的过程
    # todo 大白话——这是一个从零开始完整走一遍评估流程的测试函数，跟_evaluate_model效果一样，
    # todo 只是它自己先读数据再切分，适合手动调试验证。
    def _test_EVALUATE_MODEL(self, random_state=42, test_size=0.2):
        """评估已训练模型的性能（使用划分的验证集）
        Args:
            random_state: 随机种子，默认为42，确保结果可复现
            test_size: 验证集比例，默认为0.2（20%）
        """
        print('评估 ➡️ 已训练模型的性能')
        # 加载测试数据
        with open(Config.BERT_TRAIN_DATA, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]
        texts = [item['query'] for item in data]
        labels = [item['label'] for item in data]

        # 划分训练集和验证集（与训练时保持一致）
        _, val_texts, _, val_labels = train_test_split(
            texts,  # 完整文本
            labels,  # 完整标签
            test_size=test_size,  # 验证集比例
            random_state=random_state  # 随机种子
        )

        print(f"使用随机种子: {random_state}")
        print(f"验证集比例: {test_size}")
        print(f"验证集大小: {len(val_texts)} 条")

        # 将字符串标签转换为数字标签
        # self.label_map == {"通用知识": 0, "专业咨询": 1}
        val_labels_numeric = [self.label_map[label] for label in val_labels]

        # 评估验证集
        self._evaluate_model(val_texts, val_labels_numeric)


    # todo 7. --------意图识别API--------
    # todo 这个方法的作用是：整个分类器的"对外接口"！rag_system.py只调这一个方法。
    # todo 输入用户的一句话，输出"通用知识"或"专业咨询"。
    # todo 流程：文本→分词编码→模型推理→取最高分→翻译成中文标签。
    def predict_category(self, query):
        """用训练好的模型对单个查询进行分类, 判断它是"通用知识", 还是"专业咨询"
        :param query: 用户输入的查询文本, 例如: 什么是AI?  我的保险怎么理赔?
        :return: 意图识别的分类结果, 要么是"通用知识", 否则是"专业咨询"
        """
        # 检查已训练好的模型是否加载
        if self.model is None:
            logger.error("模型未加载")
            return '通用知识' # 后续直接调用大模型回答query
            # todo 如果模型没加载成功，兜底返回"通用知识"，让后续流程直接走大模型通用回答。

        # 对输入查询的文本进行编码
        # todo 这里是单条文本编码（不是批量的list），所以参数名是query而不是texts。
        # todo 但分词器的用法完全一样：切词→转ID→补全/截断到128。
        encoding = self.tokenizer(
            query, # 用户提问文本
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        # 将编码移到指定设备
        # todo 字典推导式：把encoding里的每个tensor都搬到GPU/CPU上。
        # todo 模型在哪个设备，数据也必须跟到哪个设备，否则会报错。
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        # print(encoding)
        '''{
            'input_ids': tensor([[101, 2769, 4638,  ...]], device='cuda:0'),  # 文本的数字ID
            'attention_mask': tensor([[1, 1, 1, ...]], device='cuda:0')       # 注意力掩码
            'token_type_ids': tensor([[0, 0, 0, ...]], device='cuda:0')       # 句子标记
        }'''

        # 模型进行预测
        # todo torch.no_grad(): 大白话——"不要计算梯度"。告诉PyTorch："我只是用模型做预测，不训练，
        # todo 所以别记录梯度信息，省显存、跑更快。"这是推理阶段的标配写法。
        with torch.no_grad(): # 直接调用模型 不计算梯度
            # 进行分类预测并输出
            # todo self.model(**encoding): 大白话——把编码好的文本喂给BERT模型做一次"前向传播"。
            # todo **encoding 是Python的解包语法，相当于 model(input_ids=..., attention_mask=..., token_type_ids=...)
            # todo 模型输出一个对象，其中outputs.logits就是分类分数（一个1×2的tensor）。
            outputs = self.model(**encoding)
            # 提取预测结果 outputs.logits是预测分数
            # todo torch.argmax(outputs.logits, dim=-1): 大白话——从两个分数里找最大值的位置。
            # todo dim=-1表示"在最后一个维度（类别维度）上找最大值"。
            # todo .item(): 大白话——把只有一个元素的tensor转成普通Python数字。
            # todo 比如tensor([1]) → 1，这样后面才能当字典的key用。
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        # 根据预测结果返回类别
        # todo reverse_label_map: 大白话——"反向对照表"，把数字转回中文。
        # todo {"通用知识": 0, "专业咨询": 1} 翻转成 {0: "通用知识", 1: "专业咨询"}
        reverse_label_map = {v: k for k, v in self.label_map.items()}
        # {"通用知识": 0, "专业咨询": 1} → {0: '通用知识', 1: '专业咨询'}
        # {0: '通用知识', 1: '专业咨询'}[0] → 通用知识
        # {0: '通用知识', 1: '专业咨询'}[1] → 专业咨询
        return reverse_label_map[prediction]

    # todo 这个方法的作用是：测试意图识别功能，用几条例子验证predict_category能不能正确分类。
    def _test_PREDICT_CATEGORY(self):
        print('测试➡️ 查询问题的意图识别')
        # 测试分类
        queries = [
            "768除以12是多少？",  # 通用知识
            "学习AI大模型课程就业薪资多少？",  # 专业咨询
            "什么是机器学习？",  # 通用知识
            "如何申请教育贷款？",  # 专业咨询
        ]
        # todo 每次测试都重新创建一个QueryClassifier实例，走一遍加载模型→初始化流程。
        query_classify = QueryClassifier()
        for q in queries:
            result = query_classify.predict_category(q)
            print(f"查询: {q}")
            print(f"分类: {result}\n")



# todo __main__块：大白话——"程序入口"。当直接运行这个文件时（python query_classifier.py），
# todo 会执行下面的测试代码。如果被别的文件import导入，则不会执行下面的代码。
if __name__ == '__main__':
    # todo 创建分类器实例，会触发__init__→_load_model()，加载模型
    query_classify = QueryClassifier()
    print('测试开始 ... ')

    '''初始化'''
    # query_classify._test_INIT()

    '''训练集 验证集'''
    # query_classify._test_GET_TRAIN_DATA_SPLIT()
    # #
    # # '''文本编码成数字(列号)'''
    # query_classify._test_PREPROCESS_DATA()
    #
    # # '''构建数据集'''
    # query_classify._test_CREATE_DATASET()
    #
    # # '''模型训练参数'''
    # query_classify._test_GET_TRAINING_ARGS()
    #
    # # '''模型训练对象'''
    # query_classify._test_GET_TRAINER()

    '''模型训练、评估'''
    # query_classify._train_model()

    '''意图识别：预测问题类别'''
    query_classify._test_PREDICT_CATEGORY()







































