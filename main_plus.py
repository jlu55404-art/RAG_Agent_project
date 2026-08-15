# todo 导入标准库：time 用来计时，traceback 用来打印详细错误堆栈
import time, traceback
# todo uuid 库用来生成全局唯一的会话 ID（类似身份证号，保证每个会话的 ID 不重复）
import uuid # 导入 UUID 库，生成唯一会话 ID

# 导入 MySQL 和 Redis 客户端，管理数据库和缓存
# todo Bm25SoftmaxSearch：基于 BM25 算法的文本搜索类，在 MySQL 里做关键词匹配，类似于全文搜索引擎
from mysql_qa.core_search import Bm25SoftmaxSearch
# 导入 RAG 系统组件，用于知识库检索和答案生成
# todo VectorStore：连接 Milvus 向量数据库，做语义相似度检索（不像 BM25 是关键词匹配，这个理解语义）
from rag_qa.core.vector_store import VectorStore
# 使用新的rag_system代码
# todo RAGSystem：完整的 RAG 流程，做"向量检索→拼接提示词→大模型生成答案"
from rag_qa.core.rag_system import RAGSystem

# todo get_llm：工厂函数，根据配置返回一个大语言模型实例（可能是本地 Ollama 或云端 API）
from rag_qa.core.llm_provider import get_llm


# 导入配置和日志工具，用于系统配置和日志记录
from base import logger, Config


# todo IntegratedQASystem 是一个"混合问答系统"：先用 MySQL BM25 快速匹配，匹配不上再走 RAG 做深度检索
class IntegratedQASystem:
    # todo __init__ 是构造函数，创建这个类的实例时会自动执行，用来初始化各种子组件
    def __init__(self):
        # 初始化mysql qa系统核心搜索函数
        # todo Bm25SoftmaxSearch：用 BM25 算法在 MySQL 题库里搜，返回最相似的问题和答案
        self.bm25_search = Bm25SoftmaxSearch()
        # 数据库客户端
        # todo mysql_client 和 redis_client 是从 bm25_search 里拿到的数据库连接对象
        self.mysql_client = self.bm25_search.mysql_client
        # todo redis_client 连接 Redis 缓存，可以用来缓存 RAG 生成的答案，下次同样问题直接返回
        self.redis_client = self.bm25_search.redis_client
        # 初始化向量存储检索对象
        # todo VectorStore 连接 Milvus，负责把问题转成向量，在向量库里找最相关的文档块
        self.vector_store = VectorStore()
        # 初始化RAG系统
        # todo RAGSystem 是 RAG 的主控类，封装了"检索→生成答案"的完整链路
        self.rag_system = RAGSystem()
        # 大模型对象
        # todo llm 是大语言模型对象，用来生成最终的答案文本
        self.llm = get_llm()
        # 初始化对话历史表
        self._init_conversation_table()

    # todo _init_conversation_table：在 MySQL 里建一张 conversations 表，用来存用户每次的问答记录
    def _init_conversation_table(self):
        """初始化MySQL中的conversations表，用于存储对话历史"""
        # todo check_and_reconnect：先检查 MySQL 连接是否还活着，断了就重新连接
        self.mysql_client.check_and_reconnect() # 重连
        try:
            # 创建 conversations 表，包含会话 ID、问题、答案和时间戳
            # todo CREATE TABLE IF NOT EXISTS：如果表不存在就创建，已存在就跳过（不会重复建表）
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS conversations(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    INDEX idx_session_id (session_id)
                ) ENGINE=InnoDB
                """
            # 执行创建表的 SQL 语句 并提交
            self.mysql_client.cur.execute(create_table_sql)
            self.mysql_client.conn.commit()
            logger.info('对话历史表创建成功')
        except Exception:
            logger.exception('对话历史表创建失败')
            raise

    # todo call_llm：调用大语言模型生成答案。注意它用 yield 而不是 return，所以是"生成器函数"
    # todo 生成器的好处是：可以一个字一个字地"流式输出"，不用等全部生成完再返回
    def call_llm(self, prompt):
        """调用llm生成答案，并流式输出
        后续FastAPI调用"""
        try:
            # 构造消息对象
            # todo messages 是 OpenAI 标准格式的消息列表，system 设定角色，user 是用户实际的问题
            messages = [
                {'role': 'system', 'content': '你是一个有用的助手'},
                {"role": "user", "content": prompt},  # 用户输入的提示
            ]
            # todo stream(messages)：以流式方式调用大模型，每生成一个字就 yield 出去
            for chunk in self.llm.stream(messages):
                # todo chunk.content 是大模型每次生成的一个文本片段（可能是一个字或几个字）
                yield chunk.content

        except Exception as e:
            logger.exception('调用LLM生成答案时出错')
            # todo 就算出错也 yield 错误信息，保证前端能收到反馈
            yield f'错误：llm调用失败 {e}'


    # todo _get_session_history：根据 session_id 从 MySQL 查最近5轮对话历史
    def _get_session_history(self, session_id):
        """获取最近5轮会话历史"""
        self.mysql_client.check_and_reconnect() # 重连
        try:
            # todo 按时间倒序查最近5条，用 fetchall() 取出所有结果
            self.mysql_client.cur.execute(
                "SELECT question, answer FROM conversations "
                "WHERE session_id = %s "
                "ORDER BY timestamp DESC LIMIT 5",
                (session_id,)
            )

            # 将查询结果转换为 [dict]
            # todo 列表推导式：把数据库查到的每一行（元组）转成字典格式
            history = [
                {'question': row[0], 'answer': row[1]}
                for row in self.mysql_client.cur.fetchall()
            ] # [Q5 Q4 Q3 Q2 Q1]（倒序的，最新的在前）
            # todo [::-1] 是 Python 切片反转技巧：把列表翻过来，变成旧→新的顺序
            return history[::-1] # [Q1 Q2 Q3 Q4 Q5]

        except Exception:
            logger.exception('获取会话历史时出错')
            return []

    # todo _update_session_history：把当前这一轮问答存入 MySQL，同时清理超过5轮的旧记录
    # todo 这是"先插入新记录→再删除旧记录"的策略，保证表中始终只保留最近5轮对话
    def _update_session_history(
            self,
            session_id: str,
            question: str,
            answer: str) -> list:
        """更新会话历史到MySQL，保留并返回最近5轮对话"""
        self.mysql_client.check_and_reconnect() # 重连
        try:
            # 插入新的对话记录
            # todo 把用户问题和系统回答插入 conversations 表，时间戳用 NOW() 取当前时间
            self.mysql_client.cur.execute("""
                    INSERT INTO conversations (
                        session_id, question, 
                        answer, timestamp
                    )
                    VALUES (%s, %s, %s, NOW())
                """, (session_id, question, answer))
            # 获取更新后的对话历史
            history = self._get_session_history(session_id)
            # 删除超出5轮的旧的会话历史记录
            # todo 这段 SQL 的意思是：保留最新的5条记录，删除该 session_id 下更早的记录
            self.mysql_client.cur.execute("""
                    DELETE FROM conversations
                    WHERE session_id = %s AND id NOT IN (
                        SELECT id FROM (
                            SELECT id
                            FROM conversations
                            WHERE session_id = %s
                            ORDER BY timestamp DESC
                            LIMIT %s
                        ) AS sub
                    )
                """, (session_id, session_id, 5))
            self.mysql_client.conn.commit() # 提交事务
            # TODO 先插再查、删除，最后提交
            #   查询到的结果会不会因为没有提交而查不到最新的插入的数据？
            # 不会！
            # 在同一个连接（同一个事务）里，你自己写进去的数据，你自己一定能看到，哪怕还没 commit。 "存档"（commit）是给别人看的
            logger.info(f'会话{session_id} 历史更新成功')
            # 返回更新后的最新的5轮对话
            return history

        except Exception:
            logger.exception(f'会话{session_id} 历史更新失败')
            # 回滚
            self.mysql_client.conn.rollback()
            raise # 抛出异常

    # todo _clear_session_history：清空指定 session_id 的所有对话记录
    # todo 返回 True 表示清除成功，False 表示失败
    def _clear_session_history(self, session_id: str) -> bool:
        """清除指定会话历史"""
        self.mysql_client.check_and_reconnect() # 重连
        try:
            # 删除 session_id 对应的会话历史记录
            self.mysql_client.cur.execute(
                "DELETE FROM conversations "
                "WHERE session_id = %s",
                (session_id,)
            )
            self.mysql_client.conn.commit()
            logger.info(f'会话{session_id} 历史清除成功')
            return True # 清除成功
        except Exception:
            logger.exception(f'会话{session_id} 历史清除失败')
            return False # 清除失败


    # todo query 是核心查询方法：先走 MySQL BM25 快速匹配，匹配不上再走 RAG 深度检索
    # todo 它的返回值是生成器（用 yield），支持流式输出答案
    def query(
        self, query, source_filter=None, session_id=None):
        """查询集成系统，支持对话历史和流式输出
        :param query: 查询问题
        :param source_filter: 学科过滤
        :param session_id: 会话id
        :return:
        """
        start_time = time.time()
        logger.info(f'用户查询: {query}，'
                    f'学科过滤条件: {source_filter}，'
                    f'会话ID: {session_id}')
        # 获取会话历史 如果有session_id正常去mysql查询获取 没有就[]
        # todo 如果有 session_id，就去查历史对话，否则用空列表
        session_history = self._get_session_history(session_id) \
            if session_id else []


        # mysql QA
        # todo bm25_search.search：①返回 (answer, need_rag) ②need_rag=True 表示 MySQL 没找到可靠答案
        answer, need_rag = self.bm25_search.search(query)
        # 如果需要RAG系统
        # todo not need_rag 表示 MySQL 已经找到了可靠答案，直接用，不需要走 RAG
        if not need_rag:
            logger.info(f'FQA系统答案长度：{len(answer)}')
            if session_id:
                # 更新会话历史
                self._update_session_history(
                    session_id, query, answer)
            processing_time = time.time() - start_time
            logger.info(f'FQA系统 ➡️ '
                        f'问题：{query}；'
                        f'答案：{answer[:30]}；'
                        f'处理时间：{processing_time:.4f}秒')
            # 一次性返回答案，True表示标记为完整答案
            # 函数中出现yield返回 就不能出现return
            # todo yield (answer, True)：True 表示这是完整答案，前端收到后可以停止等待
            yield answer, True

        # RAG系统
        # todo need_rag=True：MySQL 的 BM25 匹配分数不够高，需要用 RAG 做语义检索
        else:
            logger.info('FQA无可靠答案，切换到RAG系统')
            # 收集完整的答案字符串
            collected_answer = ''
            # todo generate_answer_withhistory：RAG 系统带历史对话的生成，逐个 chunk 流式返回
            for chunk in self.rag_system.generate_answer_withhistory(
                query, source_filter, session_history):
                collected_answer += chunk # 累积答案（把所有 chunk 拼接起来得到完整答案）
                # 流式返回答案，False表示标记为非完整答案
                # todo yield (chunk, False)：False 表示还没结束，后面还有更多内容
                yield chunk, False
            # TODO 缓存答案
            # todo 把 RAG 生成的完整答案缓存到 Redis，下次同样问题可以直接返回
            # todo 只缓存有效答案，不缓存"人工客服"兜底错误
            if collected_answer and '人工客服' not in collected_answer:
                self.redis_client.set_answer(query, collected_answer)

            # 更新会话历史记录
            if session_id:
                self._update_session_history(
                    session_id, query, collected_answer)
            processing_time = time.time() - start_time
            logger.info(f'RAG系统 ➡️ '
                        f'问题：{query}；'
                        f'答案：{collected_answer[:30]}；'
                        f'处理时间：{processing_time:.4f}秒')
            # True表示标记为完整答案
            yield '', True


# todo 系统入口
# todo main 函数：创建 IntegratedQASystem 实例，进入交互式循环，支持带历史记忆的问答
def main():
    # 创建带记忆功能的 FQA_RAG系统实例
    # todo IntegratedQASystem 集成了 MySQL BM25 快速搜索 + RAG 深度检索，还有对话历史管理
    qa_system = IntegratedQASystem()
    # 生成唯一会话ID
    # todo uuid4() 生成一个随机 UUID（类似 550e8400-e29b-41d4-a716-446655440000），确保每次会话 ID 不重复
    session_id = str(uuid.uuid4())
    # 打印欢迎提示语
    print('\n欢迎使用集成问答系统')
    print(f'会话ID：{session_id}')


    while True:
        print('输入查询进行问答，输入 "exit" 退出系统')
        query = input('\n请输入您的问题：').strip()
        if query.lower() == 'exit':
            print('感谢使用，再见！')
            break

        # 获取用户输入的学科标识
        # todo 用户输入学科类别来过滤检索范围（只检索某个学科的文档）
        source_filter = input(f"请输入学科类别 "
                              f"({'/'.join(Config.VALID_SOURCES)}) "
                              f"(直接回车默认不过滤): ").strip().lower()
        if source_filter and source_filter not in Config.VALID_SOURCES:
            print(f'无效的学科类别，将使用默认值None(不过滤)')
            source_filter = None
        if source_filter == '':
            source_filter = None
        # 打印答案的提示
        print('\n➡️ 答案 ：', end='', flush=True)
        # 调用系统查询方法，传入问题、学科过滤和会话ID
        # todo qa_system.query 返回生成器，每次 yield (chunk, is_complete)
        for chunk, is_complete in qa_system.query(query,
                                     source_filter,
                                     session_id):
            if chunk:
                # todo end='' 表示不自动换行，flush=True 表示立即输出（不缓冲），实现逐字打印效果
                print(chunk, end='', flush=True)
            # todo is_complete=True 表示答案已经输出完毕
            if is_complete:
                print() # 主动强制换行（前边输出答案都不换行！）
                break
        # 展示会话历史
        history = qa_system._get_session_history(session_id)
        print('\n最近对话历史：')
        for idx, entry in enumerate(history, 1):
            # 按顺序打印历史对话记录（序号从1开始而不是0）
            print(f'{idx}. 问题：{entry["question"]}\n '
                  f'回答：{entry["answer"]}')

# todo if __name__ == '__main__'：Python 标准入口判断，直接运行时执行 main()，被 import 时不执行
if __name__ == '__main__':
    """程序入口"""
    main()


    """"""
    # a503d94b-f138-41eb-9b81-76c977de5346







