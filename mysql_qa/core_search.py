
# todo 核心搜索模块：实现基于 BM25 + Softmax 的问答匹配
# todo BM25：一种经典的信息检索算法，统计词频和文档长度来计算查询和文档的相似度
# todo BM25Okapi：BM25 的一种常用实现版本
# todo Softmax：把一组数字转换成概率分布（所有概率加起来等于1），让最高分更突出、最低分更不显眼
import numpy as np
from rank_bm25 import BM25Okapi
# TODO 从项目根路径开始导包！
from mysql_qa.utils.mysql_client import MysqlClient
from mysql_qa.utils.redis_client import RedisClient
from mysql_qa.utils.preprocess import preprocess_text
from base import logger


# todo Bm25SoftmaxSearch 类：核心搜索引擎，用 BM25 算法计算用户问题和题库中每个问题的相似度，再用 Softmax 归一化后取最高分
class Bm25SoftmaxSearch:
    """核心搜索"""

    # todo __init__ 方法：初始化搜索引擎，连接 MySQL 和 Redis，加载题库数据，构建 BM25 模型
    # todo MysqlClient：MySQL 数据库客户端，用来存取问答数据
    # todo RedisClient：Redis 缓存客户端，用来缓存数据加快访问速度
    def __init__(
            self,
            mysql_client=MysqlClient(),
            redis_client=RedisClient()
        ):
        # todo mysql_client：MySQL 数据库连接对象，负责从数据库读写问答数据
        self.mysql_client = mysql_client
        # todo redis_client：Redis 缓存连接对象，负责缓存问题和答案，避免频繁查数据库
        self.redis_client = redis_client

        # todo original_questions：原始问题列表，比如 ["什么是Python", "如何安装MySQL", ...]
        # 原始问题列表 [问题1, 问题2,...]
        self.original_questions = None
        # todo tokenized_questions：分词后的问题列表，比如 [["什么","是","Python"],["如何","安装","MySQL"],...]
        # 分词后的问题列表 [[词1，词2...], [词1, 词2...]...]
        self.tokenized_questions = None
        # todo bm25：BM25 模型对象，传入分词后的问题列表就能构建，用来计算文本相似度
        # bm25模型
        self.bm25 = None
        # TODO 加载数据
        # todo 初始化时自动调用 _load_data 加载题库数据并构建 BM25 模型
        self._load_data()

    # todo _load_data 方法：从 Redis 缓存或 MySQL 数据库中加载题库数据，然后构建 BM25 检索模型
    # todo 数据加载顺序：先看 Redis 有没有缓存 → 没有就从 MySQL 读 → 读到后缓存到 Redis → 最后构建 BM25
    def _load_data(self):
        """ 构造bm25对象，加载数据到内存中：
        1. 判断redis中是否有数据，两key但凡有一个没数据
            就从mysql中读取数据
                mysql没数据就终止函数
                有数据就缓存到redis中
        2. 构造bm25对象"""

        # redis中 问题列表 的key名 [问题1, 问题2,...]
        original_key = 'qa_original_questions'
        # redis中 问题分词列表 的key名 [[词1,词2,..],...]
        tokenized_key = 'qa_tokenized_questions' # 问题分词之后的key

        # todo 从 Redis 缓存中取出问题列表和分词列表
        # 从redis中获取所有问题构成的列表
        self.original_questions = self.redis_client.get_data(original_key)
        # 从redis中获取所有问题分词列表
        self.tokenized_questions = self.redis_client.get_data(
            tokenized_key)

        # todo 如果 Redis 中没有缓存数据（任一个为空），就从 MySQL 数据库中读取
        # 两个key取出但凡有一个数据为空，就从mysql中获取数据
        if not self.original_questions or not self.tokenized_questions:
            # todo fetch_all_questions 从数据库查询所有问题，返回格式是 [(问题1str), (问题2str), ...]
            # todo 列表推导式把元组拆开，只取里面的字符串
            self.original_questions = [
                question_str  # 是把外层元组拆开，只取出里面的字符串
                for (question_str,)
                in self.mysql_client.fetch_all_questions()
            ] # [(问题1str), (问题2str), ...]
            # 💡调试：看一下 问题列表 的情况
            print(len(self.original_questions),
                  self.original_questions[0])

        # todo 如果 MySQL 里也没有数据（问题列表为空），直接报错返回
        # 如果mysql中没有数据，就终止函数
        if not self.original_questions:
            logger.error(f"❌️ 没有原始问题数据，请先插入数据！")
            return

        # todo 如果有原始问题但没有分词列表，就重新对每个问题进行分词
        # 如果分词列表为空，就重新对问题进行分词、构造分词列表
        if not self.tokenized_questions:
            # 👉构造问题分词列表 [[词1，词2...], [词1, 词2...]...]
            # todo preprocess_text：自定义的分词函数，把中文句子切成一个个词
            # todo is_generator=False 表示返回列表而不是生成器
            self.tokenized_questions = [
                preprocess_text(q, is_generator=False)
                for q in self.original_questions
            ]

        # todo 把问题和分词结果都缓存到 Redis，下次启动就不需要重新分词了
        # 问题列表 和 问题分词列表 都缓存到redis中
        self.redis_client.set_data(original_key, self.original_questions)
        self.redis_client.set_data(tokenized_key, self.tokenized_questions)
        logger.info(f"✅ 缓存问题数据成功：{len(self.original_questions)}")

        # todo 用分词后的问题列表构建 BM25 模型，之后可以用来计算任意查询与题库的相似度
        # 构造bm25对象
        self.bm25 = BM25Okapi(self.tokenized_questions)
        logger.info(f"✅ 构造BM25模型成功")

    # todo _test_LOAD_DATA 静态方法：测试用的，用来验证数据加载、Redis 和 MySQL 查询是否正常
    @staticmethod
    def _test_LOAD_DATA(question):
        # 测试 _load_data函数
        bm25search_obj = Bm25SoftmaxSearch()
        print("🔍数据查询功能 检查开始")
        print("query:", question)
        print("redis:", bm25search_obj.redis_client.get_answer(question))
        print("mysql:", bm25search_obj.mysql_client.fetch_answer(question))
        print("🔍数据查询功能 检查结束")
        print("🔍检查 问题列表 和 问题分词列表 中的第一条数据")
        print(bm25search_obj.original_questions[0])
        print('分词：', bm25search_obj.tokenized_questions[0])


    # todo softmax
    # todo _softmax 方法：对 BM25 得分做 Softmax 归一化，把原始分数转成 0~1 之间的概率，所有概率加起来等于 1
    # todo 大白话：BM25 算出来一堆分数有好有坏，Softmax 能把最高分"放大"、低分"压小"，让最匹配的问题更显眼
    def _softmax(self, scores):
        # todo np.exp：计算 e（自然常数 ≈2.718）的多少次方
        # todo 先减去最大值再算指数，是为了防止数字太大导致计算溢出（变无穷大）
        exp_scores = np.exp(scores-np.max(scores))
        # todo 每个指数值除以所有指数值的总和，得到各自己的概率（0~1 之间，加起来等于 1）
        return exp_scores / np.sum(exp_scores)

    # todo _test_SOFTMAX 静态方法：演示 Softmax 的计算过程，每一步都打印出来帮助理解
    @staticmethod # softmax演示函数
    def _test_SOFTMAX(scores):
        # todo 步骤拆解：SOFTMAX的原理
        print('演示➡️ softmax的过程')
        print("原始分数 scores：", scores)

        # todo 第一步：找到分数中的最大值
        max_score = np.max(scores)
        print("最大值 np.max：", max_score)

        # todo 第二步：每个分数减去最大值（防止指数计算溢出）
        scores_shifted = scores - max_score
        print("减去最大值后：", scores_shifted)

        # todo 第三步：对每个减去最大值后的分数计算 e 的多少次方
        exp_scores = np.exp(scores_shifted)  # e的多少次方
        print("指数 np.exp 后：", exp_scores)

        # todo 第四步：把所有指数值加起来
        sum_exp = np.sum(exp_scores)
        print("指数总和：", sum_exp)

        # todo 第五步：每个指数值除以总和，得到最终概率
        result = exp_scores / sum_exp
        print("最终 softmax 概率：", result)
        print("概率总和：", np.sum(result))
        return result

    # todo _test_SCORES_MAX 静态方法：演示如果不做"减去最大值"处理，大数值的指数计算会溢出变成无穷大（inf）
    @staticmethod # 演示函数
    def _test_SCORES_MAX(scores=[1000, 1001, 1002]):
        # todo 计算爆炸的演示
        # （减去最大值的作用：防止计算爆炸）
        # 计算 e^1000, e^1001, e^1002 ==> [inf inf inf]
        # todo 大白话：e 的 1000 次方是个天文数字，Python 算不出来，直接给 inf（无穷大）
        print('演示➡️ 计算爆炸的结果')
        print(np.exp(scores))


    # todo 检索
    # todo search 方法：最关键的方法，输入用户问题，返回最匹配的答案
    # todo threshold=0.85：相似度阈值，Softmax 归一化后的最高分 ≥ 0.85 才认为找到了可靠答案
    # todo 返回值：(answer, need_rag) — answer 是答案字符串，need_rag 是布尔值表示是否需要交给 RAG 系统进一步处理
    def search(self, query, threshold=0.85):
        """
        参数 query : 问题字符串
        参数 threshold=0.85 : 设置的相似度阈值
            >=0.85 相似度极高，判定命中
        """
        #防御：输入不合法直接返回
        if not query or not isinstance(query, str):
            logger.error(f"❌️ 请输入有效的问题！")
            return None, False # False 是后续不调用RAG的信号

        # todo 第一步：先查 Redis 缓存，看这个问题之前是不是问过（缓存命中就直接返回答案，不用再算一遍）
        # 第一关：redis中获取问题答案
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            logger.info(f"✅ 获取问题答案成功：{cached_answer}")
            return cached_answer, False

        # todo 第二步：Redis 没有缓存，开始用 BM25 + Softmax 计算相似度
        # 如果redis缓存中没有答案，就从mysql中获取答案
        try:
            # 1. 提问的问题进行分词
            # todo 对用户问题分词，比如 "什么是张量" → ["什么", "是", "张量"]
            query_tokens = preprocess_text(query, is_generator=False)
            # 2. 计算得分：所有问题的相关性得分
            # todo BM25 会返回一个得分数组，每个值代表用户问题和对应题库问题的相似度
            scores = self.bm25.get_scores(query_tokens)
            # 3. softmax算法对bm25得分进行归一化
            # todo Softmax 归一化，把原始得分变成概率分布，最匹配的问题概率最高
            softmax_scores = self._softmax(scores)
            # 4. 获取得分最高的问题索引，基于索引获取问题
            # todo argmax() 返回数组中最大值的位置（第几个）
            best_idx = softmax_scores.argmax() # 数组→最大值的索引
            # todo 拿到最高分是多少
            best_score = softmax_scores[best_idx]
            logger.info(f"✅ 问题: {query}, 最高分：{best_score}")

            # 5. 判断最高分是否超过阈值0.85
            # todo 如果最高分 ≥ 阈值 0.85，说明找到了高度匹配的问题，可以返回答案
            if best_score >= threshold:
                # 5.1. 获取最高分对应的原始问题完整文本
                # todo 根据索引取出原始问题文本（用来查答案）
                original_question = self.original_questions[best_idx]
                # 5.2. 从redis中获取问题对应的答案
                # todo 先从 Redis 缓存查答案（快），没有再查 MySQL（慢但数据全）
                redis_result = self.redis_client.get_answer(original_question)
                # 5.3. 如果redis有答案
                if redis_result:
                    answer = redis_result # 答案
                    logger.info(f"✅ 从redis中查询成功，答案长度：{len(answer)}")
                else:
                    # 5.4. 如果redis中没有答案，就从mysql中获取答案
                    answer = self.mysql_client.fetch_answer(original_question)
                    logger.info(f"✅ 从mysql中查询成功，答案长度：{len(answer)}")

                # todo 找到答案后，把问题和答案的映射缓存到 Redis，下次再问就不用重新查了
                if answer: # 确保有答案才缓存
                    # 5.5. 缓存答案
                    # todo 缓存两条映射关系：用原始问题和用户提问都能查到答案
                    # 原始问题和答案的映射 存入缓存
                    self.redis_client.set_answer(original_question, answer)
                    # 用户提问和答案的映射 存入缓存
                    self.redis_client.set_answer(query, answer)
                    logger.info(f"✅ 缓存答案成功：{original_question}")

                # todo 返回答案，need_rag=False 表示不需要 RAG 系统介入
                return answer, False # 返回答案，同时不调用RAG

            # todo 6. 最高分低于超过阈值0.85, 返回 None, True
            # todo 最高分没达到阈值，说明题库里没有足够匹配的问题，返回 None 和 True 表示需要 RAG 系统来回答
            logger.info(f"💡 没有可靠答案，softmax相似度最高分：{best_score:.3f}")
            return None, True # 无答案，True表示后续调用RAG

        except Exception as e:
            # todo 整个搜索过程出异常了，记录错误并返回 None, True 让 RAG 系统兜底
            logger.exception(f"❌ 获取答案失败")
            return None, True # 失败，True表示后续调用RAG


# todo 以下是直接运行本文件时的测试/演示代码
if __name__ == '__main__':
    import random

    # print('测试➡️ _load_data 加载数据')
    # q_list = [
    #     '关联子查询的执行顺序是什么', '归一化算法问题',
    #     'os.path.dirname() 方法有什么作用？',
    #     '函数和类区别', '什么是张量'
    # ]
    # q = random.choice(q_list)
    # Bm25SoftmaxSearch._test_LOAD_DATA(q)

    # print('演示➡️ 计算爆炸的结果')
    # Bm25SoftmaxSearch._test_SCORES_MAX([1000, 1001, 1002])

    # todo 测试核心搜索流程：随机选一个问题，看系统能否找到答案
    print('测试➡️ search核心逻辑')
    q_list = [
        '子查询的执行顺序是什么', '算法问题：归一化',
        'os.path.dirname()',
        '类和函数区别', '张量是什么'
    ]
    # todo random.choice 从列表中随机选一个
    q = random.choice(q_list)
    result = Bm25SoftmaxSearch().search(q)
    print(result)
