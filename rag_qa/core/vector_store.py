# 向量数据库
# todo 向量检索+重排序
# todo 大白话：这个文件的核心工作是——把用户问题扔进向量数据库（Milvus）里搜出相关文档，
# todo 然后用重排序模型对搜出来的文档进行精排，最终返回最相关的结果给用户。

import sys, torch
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值md5
# todo 大白话：hashlib 就像给每段文字拍个"指纹"，同样的文字永远生成同样的指纹，
# todo 这样就能保证同样的文档不会重复入库。
import hashlib
# 导入 Milvus 相关类
# todo 大白话：Milvus 是一个专门存"向量"的数据库。普通数据库存的是数字和文字，
# todo 向量数据库存的是"语义"，它能把意思相近的文本找出来（哪怕用词完全不一样）。
# todo MilvusClient：连接数据库的客户端；DataType：定义字段类型（整数、字符串等）；
# todo AnnSearchRequest：发起一次向量搜索请求；WeightedRanker：把两种搜索方式的结果加权合并。
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
# todo 大白话：Document 是 LangChain 框架里的一个"文档包装盒"，
# todo 里面装了两样东西：page_content（文档正文）和 metadata（元数据，比如来源、时间戳）。
from langchain_core.documents import Document
# 导入BGE-M3模型嵌入函数，用于生成文档和查询的向量
# todo 大白话：BGE-M3 是 BAAI（智源研究院）开源的一个模型，它能把一段文字同时变成两种向量——
# todo 稠密向量（捕捉语义"意思"）和稀疏向量（捕捉关键词"词频"），一个模型干两份活。
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入CrossEncoder，接入重排模型，用于重排序
# todo 大白话：CrossEncoder（交叉编码器）是重排序的核心。前面的搜索是"粗排"（海选），
# todo CrossEncoder 做的是"精排"——把问题和每个候选文档同时喂给模型，模型直接打出相关性分数，
# todo 比单纯靠向量距离算分准得多，但速度更慢，所以只对少量候选结果用。
from sentence_transformers import CrossEncoder

from base import Config, logger, get_abs_path
# 导入 process_documents文档加载、分割函数
from rag_qa.core.process_documents import process_documents


# todo 这个类是整个向量检索系统的"大管家"，负责三件事：
# todo  1. 连接和管理 Milvus 向量数据库（创建表、建索引、增删数据）
# todo  2. 调用 BGE-M3 模型把文字变成向量（向量就是一堆数字，代表文字的含义）
# todo  3. 执行混合检索（稠密+稀疏）+ 重排序，返回最相关的文档
class VectorStore:
    """milvus存储 检索"""

    def _log(self, method_name, msg):
        """统一日志格式: [vector_store.VectorStore.方法名:行号] 消息"""
        line_no = sys._getframe(1).f_lineno
        logger.info(f"[vector_store.VectorStore.{method_name}:{line_no}] {msg}")

    # todo 这个方法是 VectorStore 的初始化函数（构造函数），程序启动时自动执行，
    # todo 依次完成：连接Milvus → 加载嵌入模型 → 加载重排序模型 → 创建或打开集合
    def __init__(self,
                 host=Config.MILVUS_HOST,
                 port=Config.MILVUS_PORT,
                 database=Config.MILVUS_DATABASE,
                 collection=Config.MILVUS_COLLECTION,
                 ):

        # milvus客户端
        # todo 大白话：MilvusClient 就像你和 Milvus 数据库之间的"电话线"，
        # todo 之后所有增删改查操作都通过这个 client 对象来发指令。
        self.client = MilvusClient(
            uri=f'http://{host}:{port}',
            db_name=database
        )
        self.collection_name = collection # milvus集合名称
        # todo 大白话：collection（集合）在 Milvus 里就像 MySQL 里的"表"，
        # todo 一个集合里存一类数据，每行数据包含文本+向量+元信息。

        # 确定使用cpu还是gpu
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = Config.DEVICE
        # todo 大白话：device 决定了模型在 CPU 还是 GPU 上跑。
        # todo GPU 快几十倍但需要显卡，CPU 慢但任何电脑都能跑。
        self._log("__init__", f"Using device: {self.device}")

        # 初始化BGE-M3模型嵌入函数
        # todo 大白话：embedding_function 就是"文字转向量"的引擎。
        # todo 你把一句话扔进去，它吐出一串数字（向量），
        # todo 意思相近的话吐出的数字串也相近（在数学空间里距离近）。
        self.embedding_function = BGEM3EmbeddingFunction(
            Config.EMBEDDING_MODEL_PATH, # 模型路径
            device=self.device, # cpu or gpu
            # 如果使用gpu 就使用16位半精度稠密向量 默认fp32
            # todo 大白话：use_fp16 是"半精度"模式，数字只有正常精度的一半位数。
            # todo 好处是运算更快、显存占用更少，GPU上推荐开启；CPU上不支持，所以关掉。
            use_fp16=(self.device == "cuda")
        )

        # 初始化BGE-RERANKER 重排序模型
        # todo 大白话：reranker（重排序器）是"精排裁判"。前面的向量搜索是粗筛（比如搜出20篇），
        # todo 然后 reranker 把"问题+每篇文档"一对一对地打分，按分数从高到低重新排序，
        # todo 这样最相关的文档就被排到最前面。它比向量搜索更准，但更慢，所以只在粗筛之后用。
        self.reranker = CrossEncoder(
            Config.RERANK_MODEL_PATH, # 模型路径
            device=self.device, # cpu or gpu
            # max_length=512, # 最大长度 bge-reranker-large支持8192
        )

        # 获取嵌入模型的维度数
        # todo 大白话：dense_dim 是稠密向量的维度数（比如 1024），
        # todo 意思是每个文档会被表示为 1024 个数字的一串向量。
        # todo 维度越高，能表达的信息越丰富，但检索也更慢、存储更大。
        self.dense_dim = self.embedding_function.dim['dense']

        # 创建、加载向量库集合
        # todo 大白话：这行代码确保数据库里有我们的"表"（集合），
        # todo 如果表不存在就新建一张，如果已经存在就直接用。
        self._create_or_load_collection()


    def _test_INIT(self):
        """测试初始化"""
        print("测试 ➡️ 初始化功能")
        print(self.client.list_collections())
        print(self.device)
        # VectorStore._test_BGE_M3()
        # VectorStore._test_BGE_RERANKER_LARGE()
        print(self.dense_dim)


    def _test_BGE_M3(self):
        print("测试 ➡️ BGE-M3模型批量产生向量")
        texts = [
            "需要劳动合同、医疗诊断证明、工伤认定申请表",
            "需要身份证、户口本、结婚证",
            "需要租房合同、房东身份证"
        ]
        embeddings = self.embedding_function(texts)
        print('-' * 20, '输出的向量的细节➡️ ：')
        print('1. embeddings.keys:', embeddings.keys())
        print('2. 稠密向量数量:', len(embeddings['dense']))
        print('3. 一个稠密向量:', embeddings['dense'][0])
        print('4. 稀疏向量矩阵:', embeddings['sparse'])
        print('5. 第一个稀疏向量的COO格式:', embeddings['sparse'][0])
        print('6. 第一个稀疏向量的CSR格式:', embeddings['sparse'][0].tocsr())
        print('6. 第一个稀疏向量的CSR格式:', embeddings['sparse'][[0]])
        print('7. 第1个稀疏向量的列号列表:', embeddings["sparse"][[0]].indices)
        print('7. 第1个稀疏向量的列号列表:', embeddings["sparse"][0].tocsr().indices)
        print('8. 第1个稀疏向量的权重列表:', embeddings["sparse"][[0]].data)
        print('8. 第1个稀疏向量的权重列表:', embeddings["sparse"][0].tocsr().data)
        print('-'*20)

    def _test_BGE_RERANKER_LARGE(self):
        print("测试 ➡️ BGE-RERANKER-LARGE模型")
        """测试BGE-RERANKER-LARGE模型"""
        # 你的问题
        query = "工伤认定需要什么材料？"
        # 候选文档
        docs = [
            "需要劳动合同、医疗诊断证明、工伤认定申请表",
            "需要身份证、户口本、结婚证",
            "需要租房合同、房东身份证"
        ]
        # 构造嵌套列表，每个元素是一个列表，包含问题、文档，用于模型输入
        inputs = [[query, doc] for doc in docs]
        # 进模型计算分数
        scores = self.reranker.predict(inputs)
        # 按分数从高到低排序
        reranked = sorted(
            zip(docs, scores), # docs, scores 组成一对、一对的 ，这一对就是x
            key=lambda x: x[1], # x是被排序的每个元素，基于x[1]进行排序
            reverse=True # 大到小排序
        )
        # 输出结果
        print("重排序结果：")
        for doc, score in reranked:
            print(f"分数：{score:.2f} | 内容：{doc}")


    # 创建或加载向量库的集合
    # todo 这个方法的作用是：确保 Milvus 数据库里有一个可用的"表"（集合）。
    # todo 如果表不存在，就从头建（定义字段→创建索引→建表）；
    # todo 如果表已经存在，就直接加载到内存里使用。
    # todo 大白话理解"集合"：就像 Excel 里的一个工作表，每行是一条数据，每列是一个字段。
    def _create_or_load_collection(self):
        # 判断集合是否存在
        if not self.client.has_collection(self.collection_name):
            # 不存在就创建，先创建表字段结构对象
            # todo 大白话：schema 就是"建表语句"，定义这个集合有哪些列、每列是什么类型。
            # todo auto_id=False 意思是我们自己指定每行的ID，不让系统自动生成（方便去重）。
            # todo enable_dynamic_field=True 意思是允许插入时带一些没有提前声明的字段。
            schema = self.client.create_schema(
                auto_id=False, # 禁用自动ID
                enable_dynamic_field=True, # 启用动态字段
            )
            # 添加字段 主键ID
            # todo 大白话：id 是每条数据的"身份证号"，全靠它来唯一标识一条记录。
            # todo VARCHAR(100) 表示这个ID最长100个字符（数字或字母）。
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                max_length=100,
                is_primary=True,
            )
            # 标量 文本 字段
            # todo 大白话：text 字段存的是文档的原始文本内容，就是将来要送给大模型看的字。
            # todo 65535 个字符足够存一篇中等长度的文章了。
            schema.add_field(
                field_name='text', # 文本字段
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
            # 根据设备类型确定向量类型
            # CPU使用FLOAT_VECTOR，GPU使用FLOAT16_VECTOR
            # todo 大白话：FLOAT_VECTOR 是 32 位浮点数向量（精度高但占空间），
            # todo FLOAT16_VECTOR 是 16 位浮点数向量（精度略低但省一半空间、检索更快）。
            # todo GPU 上推荐用 FLOAT16，CPU 上只能用 FLOAT32。
            dense_type = DataType.FLOAT16_VECTOR \
                if self.device=='cuda' \
                else DataType.FLOAT_VECTOR
            # 稠密向量字段
            # todo 大白话：稠密向量（Dense Vector）是一串固定长度的数字，比如 [0.12, -0.34, 0.56, ...]，
            # todo 每个数字都有值（所以叫"稠密"）。它擅长捕捉"语义"，比如"开心"和"高兴"的向量会非常接近。
            schema.add_field(
                field_name='dense_vector',
                datatype=dense_type,
                dim=self.dense_dim, # 维度和向量模型维度一致
            )
            # 稀疏向量字段
            # todo 大白话：稀疏向量（Sparse Vector）是一大串数字，但大部分是 0，只有少数位置有值。
            # todo 它类似于传统的"关键词匹配"——每个位置代表一个词，有值代表这个词出现了。
            # todo 比如"工伤认定"这个短语，只有"工""伤""认""定"对应的位置有值，其余位置全 0。
            # todo 稠密向量管"语义相似"，稀疏向量管"关键词匹配"，两个配合使用效果最好。
            schema.add_field(
                field_name='sparse_vector',
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            # 标量 父块id
            # todo 大白话：parent_id 存的是这个子块属于哪个父块。父子块有点像书的"目录"关系——
            # todo 父块是完整的一节内容（比如 1200 字），子块是把这一节切成的小片段（比如 300 字）。
            schema.add_field(
                field_name='parent_id',
                datatype=DataType.VARCHAR,
                max_length=100,
            )
            # 标量 父块内容
            # todo 大白话：parent_content 是父块的完整文本，搜索命中子块后，
            # todo 不返回子块（太小，缺上下文），而是返回它的父块（完整，有前因后果）。
            schema.add_field(
                field_name='parent_content',
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
            # 标量 学科类别
            # todo 大白话：source 标记这篇文档属于哪个学科，比如 "ai"、"java"，方便按学科过滤检索结果。
            schema.add_field(
                field_name='source',
                datatype=DataType.VARCHAR,
                max_length=50,
            )
            # 标量 时间戳
            # todo 大白话：timestamp 记录文档入库的时间，方便排查问题或按时间排序。
            schema.add_field(
                field_name='timestamp',
                datatype=DataType.VARCHAR,
                max_length=50,
            )
            # 标量 文件名
            # todo 大白话：file_name 记录文档来源的文件名，方便追溯原始文件。
            schema.add_field(
                field_name='file_name',
                datatype=DataType.VARCHAR,
                max_length=255,
            )
            # 标量 起始页码
            # todo 大白话：page_start 记录这段内容在原文件中的起始页码（PDF等分页文档）。
            schema.add_field(
                field_name='page_start',
                datatype=DataType.INT64,
            )
            # 标量 结束页码
            # todo 大白话：page_end 记录这段内容在原文件中的结束页码。
            schema.add_field(
                field_name='page_end',
                datatype=DataType.INT64,
            )
            # 标量 章节
            # todo 大白话：chapter 记录这段内容所属的章节名称，便于按章节定位。
            schema.add_field(
                field_name='chapter',
                datatype=DataType.VARCHAR,
                max_length=255,
            )
            # 标量 版本号
            # todo 大白话：version 记录知识库版本，支持版本切换和回滚。
            schema.add_field(
                field_name='version',
                datatype=DataType.VARCHAR,
                max_length=50,
            )

            # 索引对象
            # todo 大白话：索引就像书的"目录"，没有索引要翻遍整本书找一句话。
            # todo 有了索引，数据库能快速定位到相似向量在哪，速度提升成千上万倍。
            index_params = self.client.prepare_index_params()
            # 创建稠密向量索引
            # todo 大白话：IVF_FLAT 是一种"聚类索引"。它先拿 128 个"代表"把全部向量分成 128 个簇，
            # todo 检索时只需要在最近的几个簇里找，不用遍历全部数据，速度飞起。
            # todo metric_type='IP' 意思是计算两个向量的"内积"，内积越大说明越相似。
            index_params.add_index(
                field_name='dense_vector', # 稠密向量字段
                index_name='dense_index', # 索引名称
                index_type='IVF_FLAT', # 索引类型
                params={'nlist': 128}, # 索引参数
                metric_type='IP' # 度量类型
            )
            # 稀疏向量索引
            # todo 大白话：SPARSE_INVERTED_INDEX 就是"倒排索引"，
            # todo 类似于书的"关键词索引"——哪个词出现在哪篇文章里，一查就知道。
            index_params.add_index(
                field_name='sparse_vector', # 稀疏向量字段
                index_name='sparse_index', # 索引名称
                index_type='SPARSE_INVERTED_INDEX', # 倒排索引
                metric_type="IP", # 内积
                params={"drop_ratio_build": 0.2} # 丢弃20%的低权重的词
            )
            """参数drop_ratio_build: 0.2 建索引时扔掉20%最没用的特征
             1. 一堆文章（向量），每篇有 1000 个词（维度）：
                 大部分词是停用词、不重要的小权重词（比如：的、是、和）。
                 建索引时，扔200个词权重小的词
             2. drop_ratio_build一般设为0.1~0.2，0.2是最佳实践
                 【永久丢失】一点精度，省空间、提速       
             """

            # 建集合
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema, # 字段结构
                index_params=index_params # 索引
            )
            self._log("_create_or_load_collection",
                      f'集合 {self.collection_name} 创建成功')

        else: # 集合已经存在
            self._log("_create_or_load_collection",
                      f'集合 {self.collection_name} 已存在')

        # 加载集合
        self.client.load_collection(self.collection_name)


    def _test_CREATE_OR_LOAD_COLLECTION(self):
        print("测试 ➡️ 创建、加载向量库集合")
        self._create_or_load_collection()
        # 查看集合的字段信息
        print(self.client.describe_collection(self.collection_name))

        # 查看所有的索引名
        index_list = self.client.list_indexes(self.collection_name)
        print("集合中的索引名称列表：", index_list)
        for index in index_list:
            print('*' * 30)
            print("索引名称：", index)
            # 查看具体某个索引的详细配置
            index_detail = self.client.describe_index(
                collection_name=self.collection_name,
                index_name=index
            )
            print(f"\n【{index}】索引的详细配置：", index_detail)


    # 向集合中添加数据
    # todo 这个方法的作用是：把一批文档（Document 对象列表）批量写入 Milvus 数据库。
    # todo 流程是：①提取每篇文档的文本 → ②调用 BGE-M3 生成稠密和稀疏向量 → ③组装数据 → ④写入数据库。
    # todo 注意用的是 upsert（有就更新、没有就插入），所以重复运行不会产生重复数据。
    def add_documents(self, documents:[Document]):
        # 提取文档内容
        texts = [doc.page_content for doc in documents]

        # 生成向量 BGE-M3模型生成向量
        # 输出: dict, 包含: dense(稠密向量列表), sparse(稀疏向量)
        # todo 大白话：这一步把所有文本一起扔给模型，让模型批量转向量。
        # todo 批量处理比一条一条处理快很多，因为 GPU 可以并行计算。
        embeddings = self.embedding_function(texts)
        # print(embeddings)

        # 定义存储全部数据的列表
        data_list =[]
        # 遍历每个文档Document 组装、插入
        for i, doc in enumerate(documents):
            # 基于md5(32位16进制)生成文档的唯一id
            # todo 大白话：MD5 是一种哈希算法，给任意文本生成一个 32 位的"指纹"。
            # todo 同样内容永远生成同样的 MD5，所以能防止同样的文档重复入库。
            text_hash = hashlib.md5(
                doc.page_content.encode('utf-8')).hexdigest()

            # 构造稀疏向量字典
            # todo 大白话：BGE-M3 输出的稀疏向量是 CSR 格式（一种压缩存储方式），
            # todo 这里把它转成字典格式 {词ID: 权重}，方便存入 Milvus。
            sparse_vector = {} # 一个文档的稀疏向量字典
            row = embeddings['sparse'][i].tocsr()
            indices = row.indices
            values = row.data
            # 组装稀疏向量
            # 遍历结束后 一个文档的稀疏向量 就构建成 ➡️ {token_id: weight, ...}
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value

            # 组装单条文档数据
            # todo 大白话：把文本、稠密向量、稀疏向量、元信息打包成一条记录，
            # todo 就像 Excel 里的一行，每列对应一个字段。
            data = {
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings['dense'][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata['parent_id'],
                "parent_content": doc.metadata['parent_content'],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown"),
                "file_name": doc.metadata.get("file_name", ""),
                "page_start": doc.metadata.get("page_start", 0),
                "page_end": doc.metadata.get("page_end", 0),
                "chapter": doc.metadata.get("chapter", ""),
                "version": doc.metadata.get("version", ""),
            }
            data_list.append(data) # 添加单条数据到总数据列表

        # 如果data_list不是[] 就写入milvus数据库
        if data_list:
            # upsert = insert + update
            # todo 大白话：upsert 是 "insert or update" 的缩写，
            # todo 如果 ID 已存在就覆盖更新，ID 不存在就插入新数据。
            # todo 这样重复运行入库代码不会产生重复数据。
            self.client.upsert(self.collection_name, data_list)
            self._log("add_documents",
                      f"已插入 {len(data_list)} 条数据到集合 {self.collection_name}")

    def _test_ADD_DOCUMENTS(self):
        print("测试 ➡️ 文档入库")
        # 加载并分割文档，获取所有子块构成的列表 [Document]
        documents = process_documents(
            directory_path=Config.DATA_DIR)
        print('子块文档数：', len(documents))
        # 添加数据
        self.add_documents(documents)

    # ========混合检索（粗排）+ 重排序（精排）=========
    # 混合检索
    # todo 这个方法是整个检索流程的核心，分三大步：
    # todo 第一步"混合检索"：同时用稠密向量（语义）和稀疏向量（关键词）去库里面搜，加权合并结果
    # todo 第二步"提取父文档"：搜到的都是子块，去重后提取它们所属的完整父文档
    # todo 第三步"重排序"：用 BGE-Reranker 对父文档精排，返回最相关的 Top-M 篇
    # todo 大白话理解"混合检索"：稠密向量能理解"苹果手机"和"iPhone"是一个意思（语义），
    # todo 稀疏向量能精确匹配"苹果"这个词出现过（关键词），两者结合取长补短。
    def hybrid_search_with_rerank(
            self,
            query,
            k=Config.RETRIEVAL_K,
            source_filter=None
        ):
        """该函数用于执行混合检索(稠密 + 稀疏向量) + 结果重排序, 返回精准父文档
        :param query: 用户查询文本, 例如: AI学科的课程内容是什么?
        :param k: 混合检索返回的Top-K子块数量, 默认从Config.RETRIEVAL_K读取
        :param source_filter: 学科过滤条件, 例如: "ai", 仅检索ai学科的文档, None:不过滤
        :return: 重排序后的Top-M父文档列表
        """
        # 生成查询文本的向量(BEG-M3) {dense: [], sparse: []}
        # todo 大白话：把用户问题也变成向量，因为只有同在一个"向量空间"里才能比较距离。
        query_embedding = self.embedding_function([query])
        # 提取稠密向量
        # todo 大白话：dense_query_vector 是问题的语义表示（一串数字），
        # todo 要拿去和库里的所有文档的稠密向量算相似度。
        dense_query_vector = query_embedding['dense'][0]
        # 提取稀疏向量
        # todo 大白话：sparse_query_vector 是问题的关键词表示（{词ID: 权重}的字典），
        # todo 要拿去和库里的所有文档的稀疏向量算匹配度。
        sparse_query_vector = {}
        row = query_embedding['sparse'][0].tocsr()
        # row.indices是索引列表
        # row.data是权重列表
        for idx, value in zip(row.indices, row.data):
            sparse_query_vector[idx] = value

        # 过滤表达式字符串：按学科过滤
        # todo 大白话：如果传入了学科过滤条件（如 "ai"），就只在 ai 学科的文档里搜。
        # todo 不传就搜全部学科。这就像图书馆里只在"计算机"书架找书一样。
        filter_expr = f"source == '{source_filter}'" if source_filter else ""

        # 构建稠密向量检索请求对象
        # todo 大白话：AnnSearchRequest 是一个"搜索订单"，告诉 Milvus：
        # todo "我要用这个向量在 dense_vector 字段里搜，返回最像的 k 个结果"。
        # todo nprobe 从 Config.NPROBE 读取，表示搜索时探查多少个最近的聚类簇，
        # todo nprobe 越大越准但也越慢。
        dense_request = AnnSearchRequest(
            data=[dense_query_vector], # 查询向量
            anns_field='dense_vector', # milvus集合中稠密向量字段名
            # nprobe 从Config读取，表示探查簇数量
            # metric_type: "IP" 表示使用内积作为度量类型
            param={"metric_type": "IP",
                   "params": {"nprobe": Config.NPROBE}},
            limit=k, # 返回的Top-K子块数量
            expr=filter_expr, # 过滤表达式字符串：按学科过滤
        )

        # 构建稀疏向量检索请求对象
        # todo 大白话：和稠密请求类似，但是用稀疏向量去搜，匹配关键词层面的相似度。
        # todo drop_ratio_search 从 Config.DROP_RATIO_SEARCH 读取，
        # todo 搜索时丢弃低权重的词项，加速检索。
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector], # 查询向量，是稀疏的
            anns_field='sparse_vector', # milvus集合中稀疏向量字段名
            param={"metric_type": "IP",
                   "params": {"drop_ratio_search": Config.DROP_RATIO_SEARCH}},
            limit=k, # 返回的Top-K子块数量
            expr=filter_expr, # 过滤表达式字符串：按学科过滤
        )

        # 创建加权排序器
        # 稠密向量-语义权重、稀疏向量-关键词权重均从Config读取
        ranker = WeightedRanker(Config.DENSE_WEIGHT, Config.SPARSE_WEIGHT)

        # 定义输出字段列表：包含所有标量字段
        output_fields = [
            "text", "parent_id", "parent_content", "source", "timestamp",
            "file_name", "page_start", "page_end", "chapter", "version"
        ]

        # 执行hybrid混合检索
        # todo 大白话：hybrid_search 同时执行稠密和稀疏两种搜索，然后用 WeightedRanker 加权合并结果。
        # todo 权重从 Config.DENSE_WEIGHT 和 Config.SPARSE_WEIGHT 读取。
        results = self.client.hybrid_search(
            collection_name=self.collection_name, # 集合名
            reqs=[dense_request, sparse_request], # 检索请求对象列表
            ranker=ranker,
            limit=k, # 返回的Top-K子块数量
            output_fields=output_fields
            )

        self._log("hybrid_search_with_rerank",
                  f"混合检索完成, 检索到 {sum(len(hits) for hits in results)} 条子块")

        # 将检索结果转换为Document对象列表, 目的: 统一格式, 方便后续重排
        topK_documents = [VectorStore._doc_from_hit(hit) for hit in results]

        # 通过子块文档 去提取父文档的文本内容 去重、构造并返回最终的父文档
        # parent_docs ==> [Document]
        # todo 大白话：搜出来的 k 个子块可能属于同一个父块（比如 3 个子块属于同一篇长文），
        # todo 这里按 parent_id 去重，只保留唯一的父文档，避免重复内容浪费后续处理资源。
        parent_docs = VectorStore._get_unique_parent_docs(topK_documents)

        # 判断 当父文档数 < 最终需要的数量时，跳过重排序，直接返回
        # todo 大白话：如果去重后的父文档数量本来就很少（少于目标数），
        # todo 就没必要重排了，直接返回所有结果。
        if len(parent_docs) < Config.CANDIDATE_M:
            self._log("hybrid_search_with_rerank",
                      f"父文档数({len(parent_docs)}) < CANDIDATE_M({Config.CANDIDATE_M}), 跳过重排序")
            return parent_docs

        # 进入重排序阶段
        # todo 大白话：重排序是"精排"——把每个父文档和用户问题配对，让 BGE-Reranker 逐对打分，
        # todo 分数高的排在前面。这个比向量搜索更准，但计算量大，所以只在候选集上进行。
        # 构造 查询-文档 配对的列表
        pairs = [[query, doc.page_content] for doc in parent_docs]
        # 使用BEG-Reranker进行重排序，获取分数
        scores = self.reranker.predict(pairs)
        # 根据分数对父文档进行排序
        ranked_parent_docs = [
            doc for _, doc in
            sorted(zip(scores, parent_docs), reverse=True)
        ]
        # 最终返回TOP_M个父文档
        # todo 大白话：只返回分数最高的前 M 篇父文档。
        # todo M 在配置中设置（CANDIDATE_M），太大浪费 LLM token，太小信息不够。
        final_docs = ranked_parent_docs[:Config.CANDIDATE_M]
        self._log("hybrid_search_with_rerank",
                  f"重排序完成, 返回 {len(final_docs)} 篇父文档")
        return final_docs


    @staticmethod
    # todo 这个方法的作用是：把 Milvus 返回的原始检索结果（hit 命中）转换成 LangChain 的 Document 对象。
    # todo 因为后续重排序等流程期望统一的 Document 格式，所以需要一个"翻译层"。
    def _doc_from_hit(hit):
        '''将检索的一个hit命中结果构造Document对象'''
        doc_list = []
        for ret in hit:
            hit_entity = ret['entity']
            # todo 大白话：把 Milvus 返回的字段（text、parent_id 等）映射到 LangChain 的 Document 对象上。
            doc = Document(
                page_content=hit_entity['text'],
                metadata={
                    "parent_id": hit_entity['parent_id'],  # 父文档id
                    # 父文档文本
                    "parent_content": hit_entity['parent_content'],
                    "source": hit_entity['source'],  # 学科
                    "timestamp": hit_entity['timestamp'],  # 时间戳
                    "file_name": hit_entity.get('file_name', ''),  # 文件名
                    "page_start": hit_entity.get('page_start', 0),  # 起始页码
                    "page_end": hit_entity.get('page_end', 0),  # 结束页码
                    "chapter": hit_entity.get('chapter', ''),  # 章节
                    "version": hit_entity.get('version', ''),  # 版本号
                }
            )
            doc_list.append(doc)
        return doc_list

    @staticmethod
    # todo 这个方法的作用是：从子块列表中提取唯一的父文档（按 parent_id 去重），
    # todo 并用父块内容（而非子块内容）构造 Document 返回。
    # todo 大白话：搜出来一堆"碎片"（子块），但真正给大模型看的是"整块"（父块），
    # todo 而且要确保同样的父块只出现一次，不浪费 token。
    def _get_unique_parent_docs(topK_documents):
        """从子块列表中提取全部父文档的文本内容，
        去重、构造并返回[Document]"""
        parent_ids = set() # 父文档id集合，用于去重
        # todo 大白话：set（集合）就像一袋弹珠，同样的东西只放得进去一个，天然去重。
        unique_docs = [] # 去重后的父文档列表

        for docs in topK_documents:
            for doc in docs:
            # print(len(doc), doc[0])
                parent_id = doc.metadata.get('parent_id', None) # 父文档id
                # 判断父文档id不在集合中，就加入结果 TODO 这里自动去重
                # todo 大白话：如果这个 parent_id 还没出现过（不在 set 里），
                # todo 就把它加进去，并用父块内容构造一个 Document 返回。
                if parent_id and (parent_id not in parent_ids):
                    parent_ids.add(parent_id) # 加入父文档id集合
                    # 获取父文档文本内容 ，注意这里的doc还是子块文档
                    parent_content = doc.metadata.get('parent_content')
                    # 构造父文档对象
                    # todo 大白话：这里新建了一个 Document，但 page_content 换成完整的父块文本，
                    # todo 这样送给大模型的是有完整上下文的段落，而不是孤立的小片段。
                    parent_doc = Document(
                        page_content=parent_content, # 文本内容
                        metadata=doc.metadata # 父文档直接使用子文档的元数据
                    )
                    unique_docs.append(parent_doc)
        # 最后返回去重后的父文档列表
        return unique_docs


    def _test_HYBRID_SEARCH_WITH_RERANK(self, query):
        print(f'测试 ➡️ 混合检索 + 重排序: {query}')
        ret = self.hybrid_search_with_rerank(query)
        print(f'返回结果数量: {len(ret)}')
        for doc in ret:
            print('-'*20)
            print(doc.__dict__.keys())
            print(doc.metadata.keys())
            print(doc.page_content[:50])



if __name__ == '__main__':
    # VectorStore()._test_INIT()
    # VectorStore()._test_BGE_M3()
    # VectorStore()._test_BGE_RERANKER_LARGE()
    # VectorStore()._test_CREATE_OR_LOAD_COLLECTION()

    vector_store = VectorStore()
    vector_store._test_ADD_DOCUMENTS()

    print('测试 ➡️ 混合检索')
    vector_store = VectorStore()
    vector_store._test_HYBRID_SEARCH_WITH_RERANK('大模型课程介绍')
    vector_store._test_HYBRID_SEARCH_WITH_RERANK('语言模型评估指标')
