
import time
import inspect
import pymysql
from rag_qa.core.prompts import RAGPrompts  # 提示模板
from rag_qa.core.intent_recognizer import IntentRecognizer  # 意图识别器(规则匹配+LLM兜底)
from rag_qa.core.strategy_selector import StrategySelector  # 策略选择器(4种策略)
from rag_qa.core.llm_provider import get_llm  # 统一LLM提供
from rag_qa.core.vector_store import VectorStore  # 向量库混合检索
from base import logger, Config  # 日志和配置


class RAGSystem:
    """RAG系统核心类：封装意图识别、检索、答案生成、问答记录、高频缓存等完整流程"""

    def __init__(self):
        """初始化RAG系统所需组件"""
        # 向量库对象
        self.vector_store = VectorStore()
        # 基础提示词模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        # 意图识别器: 规则匹配+LLM兜底，判断用户查询是 professional 还是 general
        self.intent_recognizer = IntentRecognizer()
        # 检索策略选择器: 用于根据用户查询选择最合适的检索策略
        # 直接检索, 子查询检索, HyDE检索, 回溯检索
        self.strategy_selector = StrategySelector()
        # 大模型对象，用于:
        #   生成简化问题 子查询问题 假设答案 直接回答
        self.llm = get_llm()
        logger.info(f"[rag_system.RAGSystem.__init__:{inspect.currentframe().f_lineno}] RAG系统初始化完成")


    # ==================== 检索策略实现 ====================

    def _retrieve_directly(self, query, source_filter):
        """直接检索: 用用户查询调用向量库的hybrid_search_with_rerank混合检索, 返回结果
        :param query: 用户的原始查询文本
        :param source_filter: 检索来源过滤条件, 例如'management'→表示只检索和管理学相关的文档.
        :return: list[Document]
        """
        logger.info(f"[rag_system.RAGSystem._retrieve_directly:{inspect.currentframe().f_lineno}] 直接检索: {query}")
        try:
            docs = self.vector_store.hybrid_search_with_rerank(
                query,
                source_filter=source_filter
            )
            logger.info(f"[rag_system.RAGSystem._retrieve_directly:{inspect.currentframe().f_lineno}] 检索文档的数量: {len(docs)}")
            for i, doc in enumerate(docs):
                content_preview = doc.page_content[:150].replace('\n', ' ')
                logger.info(f"[rag_system.RAGSystem._retrieve_directly] 文档[{i}]: {content_preview}...")
            return docs
        except Exception:
            logger.exception(f"[rag_system.RAGSystem._retrieve_directly:{inspect.currentframe().f_lineno}] 直接检索失败: {query}")
            return []

    def _test_RETRIEVE_DIRECTLY(
            self,
            query="管理学中组织的基本职能有哪些?",
            source_filter="management"
    ):
        print('测试 ➡️ 直接检索策略：')
        ret = self._retrieve_directly(
            query=query, source_filter=source_filter)
        print(ret)


    def _retrieve_backtracking(self, query, source_filter):
        """使用【回溯问题】提示词模板，向大模型提问，简化问题，再用简化的问题检索文档
        :param query: 用户的原始查询文本
        :param source_filter: 检索来源过滤条件
        :return: list[Document]
        """
        logger.info(f"[rag_system.RAGSystem._retrieve_backtracking:{inspect.currentframe().f_lineno}] 回溯检索: {query}")
        prompt = RAGPrompts.backtracking_prompt().format(query=query)

        try:
            new_query = self.llm.invoke(prompt).content.strip()
            logger.info(f"[rag_system.RAGSystem._retrieve_backtracking:{inspect.currentframe().f_lineno}] 简化后的问题: {new_query}")

            docs = self.vector_store.hybrid_search_with_rerank(
                query=new_query,
                source_filter=source_filter
            )
            logger.info(f"[rag_system.RAGSystem._retrieve_backtracking:{inspect.currentframe().f_lineno}] 检索文档的数量: {len(docs)}")
            return docs

        except Exception:
            logger.exception(f"[rag_system.RAGSystem._retrieve_backtracking:{inspect.currentframe().f_lineno}] 回溯检索失败: {query}")
            return []

    def _test_RETRIEVE_BACKTRACKING(
            self,
            query="管理学中组织的基本职能有哪些?",
            source_filter="management"
        ):
        print('测试 ➡️ 回溯问题策略：')
        ret = self._retrieve_backtracking(
            query=query, source_filter=source_filter)
        print(ret)


    def _retrieve_subqueries(self, query, source_filter):
        """使用【子查询检索】提示词模板，向大模型提问，获取多个子问题
            分别向量检索后，合并去重、返回
        :param query: 用户的原始查询文本
        :param source_filter: 检索来源过滤条件
        :return: list[Document]
        """
        logger.info(f"[rag_system.RAGSystem._retrieve_subqueries:{inspect.currentframe().f_lineno}] 子查询检索: {query}")
        prompt = RAGPrompts.subquery_prompt().format(query=query)
        try:
            new_querys = self.llm.invoke(prompt).content.strip()
            new_querys = [
                q.strip() for q in new_querys.split('\n')]
            logger.info(f"[rag_system.RAGSystem._retrieve_subqueries:{inspect.currentframe().f_lineno}] 多个子问题: {new_querys}")

            if not new_querys:
                logger.error(f"[rag_system.RAGSystem._retrieve_subqueries:{inspect.currentframe().f_lineno}] 子查询生成无效，原问题：{query}")
                return []

            all_docs = []
            for query in new_querys:
                docs = self.vector_store.hybrid_search_with_rerank(
                    query=query,
                    source_filter=source_filter
                )
                logger.info(f"[rag_system.RAGSystem._retrieve_subqueries:{inspect.currentframe().f_lineno}] 检索文档的数量: {len(docs)}")
                all_docs.extend(docs)
            logger.info(f"[rag_system.RAGSystem._retrieve_subqueries:{inspect.currentframe().f_lineno}] 所有子查询完成，共计{len(all_docs)}条文档")

            unique_docs_dict = {
                doc.page_content: doc
                for doc in all_docs
            }
            unique_docs = list(unique_docs_dict.values())
            logger.info(f"[rag_system.RAGSystem._retrieve_subqueries:{inspect.currentframe().f_lineno}] 去重后的文档数量: {len(unique_docs)}")
            return unique_docs

        except Exception:
            logger.exception(f"[rag_system.RAGSystem._retrieve_subqueries:{inspect.currentframe().f_lineno}] 子查询检索失败: {query}")
            return []


    def _test_RETRIEVE_SUBQUERIES(self):
        print('测试 ➡️ 子查询策略：')
        self._retrieve_subqueries(
            query="泰勒科学管理理论和法约尔行政管理理论有什么区别?",
            source_filter="management"
        )


    def _retrieve_hyde(self, query, source_filter):
        """使用【HyDE策略(生成假设答案, 用假设答案检索文档)】提示词模板，向大模型提问
            用大模型回答再去混合检索文档
        :param query: 用户的原始查询文本
        :param source_filter: 检索来源过滤条件
        :return: list[Document]
        """
        logger.info(f"[rag_system.RAGSystem._retrieve_hyde:{inspect.currentframe().f_lineno}] hyde检索: {query}")
        prompt = RAGPrompts.hyde_prompt().format(query=query)

        try:
            hyde_answer = self.llm.invoke(prompt).content.strip()
            logger.info(f"[rag_system.RAGSystem._retrieve_hyde:{inspect.currentframe().f_lineno}] hyde答案: {hyde_answer}")

            docs = self.vector_store.hybrid_search_with_rerank(
                query=hyde_answer,
                source_filter=source_filter
            )
            logger.info(f"[rag_system.RAGSystem._retrieve_hyde:{inspect.currentframe().f_lineno}] 检索文档的数量: {len(docs)}")
            return docs

        except Exception:
            logger.exception(f"[rag_system.RAGSystem._retrieve_hyde:{inspect.currentframe().f_lineno}] hyde检索失败: {query}")
            return []

    def _test_RETRIEVE_HYDE(self):
        print('测试 ➡️ HyDE策略：')
        self._retrieve_hyde(
            query="管理学中激励理论有哪些主要流派?",
            source_filter="management"
        )


    # ==================== 检索统一入口 ====================

    def _retrieve_entry(
            self,
            query,
            source_filter=None,
            strategy=None
        ):
        """检索策略执行的统一入口
        根据检索策略调用对应的检索方法，返回([Document], strategy_name)
        :param query: 用户的查询文本.
        :param source_filter: 检索来源过滤条件.
        :param strategy: None|直接检索|回溯问题检索|子查询检索|假设问题检索
        :return: tuple(list[Document], str) 最终上下文文档列表 + 策略名称
        """
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        if strategy == '回溯问题检索':
            ranked_docs = self._retrieve_backtracking(
                query, source_filter)
        elif strategy == '子查询检索':
            ranked_docs = self._retrieve_subqueries(
                query, source_filter)
        elif strategy == '假设问题检索':
            ranked_docs = self._retrieve_hyde(
                query, source_filter)
        else:
            ranked_docs = self._retrieve_directly(
                query, source_filter)
        logger.info(f"[rag_system.RAGSystem._retrieve_entry:{inspect.currentframe().f_lineno}] 检索策略: {strategy}，检索文档数量: {len(ranked_docs)}")

        final_ranked_docs = ranked_docs[:Config.CANDIDATE_M]
        return final_ranked_docs, strategy

    def _test_RETRIEVE_ENTRY(self):
        print('测试 ➡️ 检索并合并:')
        docs, strategy = self._retrieve_entry(
            query="管理学就业方向?",
            source_filter="management",
        )
        print(f"策略: {strategy}, 文档数: {len(docs)}")


    # ==================== 高频问答缓存 ====================

    def _check_hot_questions(self, query):
        """查询高频问答缓存表，如果命中且hit_count>=3，返回缓存答案；否则返回None
        :param query: 用户查询文本
        :return: 缓存的答案字符串 或 None
        """
        conn = None
        try:
            conn = pymysql.connect(
                host=Config.MYSQL_HOST,
                port=Config.MYSQL_PORT,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
            )
            cursor = conn.cursor()
            # 模糊匹配：查询question字段包含用户查询中的关键词，按hit_count降序
            cursor.execute(
                "SELECT question, answer, hit_count FROM hot_questions "
                "WHERE question LIKE %s AND hit_count >= 3 "
                "ORDER BY hit_count DESC LIMIT 1",
                (f"%{query}%",)
            )
            row = cursor.fetchone()
            cursor.close()

            if row:
                logger.info(f"[rag_system.RAGSystem._check_hot_questions:{inspect.currentframe().f_lineno}] 命中高频缓存: hit_count={row['hit_count']}, question='{row['question']}'")
                # 更新命中次数
                try:
                    update_cursor = conn.cursor()
                    update_cursor.execute(
                        "UPDATE hot_questions SET hit_count = hit_count + 1 WHERE question = %s",
                        (row['question'],)
                    )
                    conn.commit()
                    update_cursor.close()
                except Exception:
                    logger.exception(f"[rag_system.RAGSystem._check_hot_questions:{inspect.currentframe().f_lineno}] 更新高频缓存hit_count失败")
                return row['answer']

            return None

        except Exception:
            logger.exception(f"[rag_system.RAGSystem._check_hot_questions:{inspect.currentframe().f_lineno}] 查询高频缓存失败: {query}")
            return None
        finally:
            if conn:
                conn.close()

    def _update_hot_questions(self, question, answer):
        """问答完成后，检查同一问题被问次数，如果>=3次则写入/更新hot_questions表
        :param question: 用户问题
        :param answer: 系统生成的答案
        """
        conn = None
        try:
            conn = pymysql.connect(
                host=Config.MYSQL_HOST,
                port=Config.MYSQL_PORT,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
            )
            cursor = conn.cursor()
            # 先查看qa_records中同一问题被问了多少次（精确匹配）
            cursor.execute(
                "SELECT COUNT(*) AS cnt FROM qa_records WHERE question = %s",
                (question,)
            )
            row = cursor.fetchone()
            count = row['cnt'] if row else 0

            if count >= 3:
                # 检查hot_questions中是否已存在
                cursor.execute(
                    "SELECT id, hit_count FROM hot_questions WHERE question = %s",
                    (question,)
                )
                existing = cursor.fetchone()
                if existing:
                    # 已存在，更新hit_count和answer
                    cursor.execute(
                        "UPDATE hot_questions SET hit_count = %s, answer = %s WHERE id = %s",
                        (existing['hit_count'] + 1, answer, existing['id'])
                    )
                else:
                    # 不存在，插入新记录
                    cursor.execute(
                        "INSERT INTO hot_questions (question, answer, hit_count) VALUES (%s, %s, %s)",
                        (question, answer, count)
                    )
                conn.commit()
                logger.info(f"[rag_system.RAGSystem._update_hot_questions:{inspect.currentframe().f_lineno}] 高频缓存已更新: question='{question}', count={count}")

            cursor.close()

        except Exception:
            logger.exception(f"[rag_system.RAGSystem._update_hot_questions:{inspect.currentframe().f_lineno}] 更新高频缓存失败: {question}")
        finally:
            if conn:
                conn.close()


    # ==================== 问答记录 ====================

    def _record_qa(self, user_id, session_id, question, answer, source, strategy, retrieval_count, processing_time):
        """将问答记录写入MySQL qa_records表
        :param user_id: 用户ID（可选）
        :param session_id: 会话ID（可选）
        :param question: 用户问题
        :param answer: 系统答案
        :param source: 来源标识（如 'RAG', 'LLM'）
        :param strategy: 检索策略名称
        :param retrieval_count: 检索命中文档数
        :param processing_time: 处理耗时（秒）
        """
        conn = None
        try:
            conn = pymysql.connect(
                host=Config.MYSQL_HOST,
                port=Config.MYSQL_PORT,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                charset='utf8mb4',
            )
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO qa_records "
                "(user_id, session_id, question, answer, source, strategy, retrieval_count, processing_time) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (user_id, session_id, question, answer, source, strategy, retrieval_count, processing_time)
            )
            conn.commit()
            cursor.close()
            logger.info(f"[rag_system.RAGSystem._record_qa:{inspect.currentframe().f_lineno}] 问答记录已写入: source={source}, strategy={strategy}")

        except Exception:
            logger.exception(f"[rag_system.RAGSystem._record_qa:{inspect.currentframe().f_lineno}] 写入问答记录失败: {question}")
        finally:
            if conn:
                conn.close()


    # ==================== RAG对外核心接口 ====================

    def generate_answer(self, query, source_filter=None, user_id=None, session_id=None):
        """RAG系统对外核心接口:
            接收用户查询, 完成 "高频缓存检查→意图识别→策略选择→文档检索→答案生成→问答记录→高频缓存更新" 全流程
        :param query: 用户的原始查询文本
        :param source_filter: 检索来源过滤条件, 仅对professional生效
        :param user_id: 用户ID（可选，默认None），用于问答记录
        :param session_id: 会话ID（可选，默认None），用于问答记录
        :return: 生成的最终答案
        """
        logger.info(f"[rag_system.RAGSystem.generate_answer:{inspect.currentframe().f_lineno}] 用户查询: {query}，学科过滤条件: {source_filter}")
        start_time = time.time()

        # Step 1: 高频问答缓存检查
        cached_answer = self._check_hot_questions(query)
        if cached_answer:
            processing_time = time.time() - start_time
            logger.info(f"[rag_system.RAGSystem.generate_answer:{inspect.currentframe().f_lineno}] 命中高频缓存，直接返回: 耗时={processing_time:.4f}秒")
            self._record_qa(user_id, session_id, query, cached_answer, 'FAQ', '缓存命中', 0, processing_time)
            return cached_answer

        # Step 2: 意图识别
        intent = self.intent_recognizer.recognize(query)

        # general → LLM直接回答（context为空）
        if intent == 'general':
            logger.info(f"[rag_system.RAGSystem.generate_answer:{inspect.currentframe().f_lineno}] 意图: general，直接调用大模型")
            prompt_input = self.rag_prompt.format(
                question=query,
                context='',
                phone=Config.CUSTOMER_SERVICE_PHONE,
            )
            try:
                answer = self.llm.invoke(prompt_input).content.strip()
            except Exception:
                logger.exception(f"[rag_system.RAGSystem.generate_answer:{inspect.currentframe().f_lineno}] general-大模型生成答案异常，问题：{query}")
                answer = f'抱歉, 处理您的问题时出错, 请联系人工客服: {Config.CUSTOMER_SERVICE_PHONE}'

            processing_time = time.time() - start_time
            logger.info(f"[rag_system.RAGSystem.generate_answer:{inspect.currentframe().f_lineno}] 意图: general，答案: {answer}，耗时: {processing_time:.4f}秒")
            self._record_qa(user_id, session_id, query, answer, 'LLM', '直接回答', 0, processing_time)
            self._update_hot_questions(query, answer)
            return answer

        # professional → 执行完整RAG流程
        logger.info(f"[rag_system.RAGSystem.generate_answer:{inspect.currentframe().f_lineno}] 意图: professional，执行完整RAG流程")
        context_docs, strategy = self._retrieve_entry(query, source_filter)
        retrieval_count = len(context_docs) if context_docs else 0

        # 构造上下文 [Document] → str
        if context_docs:
            context_list = [doc.page_content for doc in context_docs]
            context = '\n\n'.join(context_list)
            logger.info(f"[rag_system.RAGSystem.generate_answer:{inspect.currentframe().f_lineno}] 上下文文档数量: {len(context_docs)}")
        else:
            context = ''
            logger.warning(f"[rag_system.RAGSystem.generate_answer:{inspect.currentframe().f_lineno}] 没有找到相关文档，返回空字符串")

        # 构造提示词
        prompt = self.rag_prompt.format(
            question=query,
            context=context,
            phone=Config.CUSTOMER_SERVICE_PHONE,
        )
        logger.info(f"[rag_system.RAGSystem.generate_answer:{inspect.currentframe().f_lineno}] 提示词长度: {len(prompt)}")

        # 大模型生成答案
        try:
            answer = self.llm.invoke(prompt).content.strip()
        except Exception:
            logger.exception(f"[rag_system.RAGSystem.generate_answer:{inspect.currentframe().f_lineno}] professional-大模型生成答案异常，问题：{query}")
            answer = f'抱歉, 处理您的问题时出错, 请联系人工客服: {Config.CUSTOMER_SERVICE_PHONE}'

        processing_time = time.time() - start_time
        logger.info(f"[rag_system.RAGSystem.generate_answer:{inspect.currentframe().f_lineno}] 意图: professional，答案: {answer}，耗时: {processing_time:.4f}秒")

        # 问答记录 + 高频缓存更新
        self._record_qa(user_id, session_id, query, answer, 'RAG', strategy, retrieval_count, processing_time)
        self._update_hot_questions(query, answer)
        return answer

    def _test_GENERATE_ANSWER(self):
        print('测试 ➡️ RAG主逻辑:')
        answer = self.generate_answer(
            query="管理学中马斯洛需求层次理论有哪些内容?",
            source_filter='management'
        )
        print(answer)


    # ==================== 流式生成接口 ====================

    def generate_answer_withhistory(self, query, source_filter=None, session_history=None, user_id=None, session_id=None):
        """RAG系统流式生成接口（带历史对话上下文）
        :param query: 用户的原始查询文本
        :param source_filter: 检索来源过滤条件
        :param session_history: 历史对话列表，格式如 [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
        :param user_id: 用户ID（可选，默认None），用于问答记录
        :param session_id: 会话ID（可选，默认None），用于问答记录
        :yield: 答案的文本片段（chunk），逐块返回给调用方
        """
        logger.info(f"[rag_system.RAGSystem.generate_answer_withhistory:{inspect.currentframe().f_lineno}] 流式生成查询: {query}，历史对话条数: {len(session_history) if session_history else 0}")
        start_time = time.time()
        # 每次查询前重置来源信息，防止返回上一次请求的旧数据
        self._last_sources = []

        # 构造历史对话上下文字符串
        history_context = ''
        if session_history:
            history_lines = []
            for msg in session_history:
                role = '用户' if msg.get('role') == 'user' else '助手'
                history_lines.append(f"{role}: {msg.get('content', '')}")
            history_context = '\n'.join(history_lines)

        # Step 1: 高频问答缓存检查
        cached_answer = self._check_hot_questions(query)
        if cached_answer:
            processing_time = time.time() - start_time
            logger.info(f"[rag_system.RAGSystem.generate_answer_withhistory:{inspect.currentframe().f_lineno}] 命中高频缓存，直接返回: 耗时={processing_time:.4f}秒")
            self._record_qa(user_id, session_id, query, cached_answer, 'FAQ', '缓存命中', 0, processing_time)
            yield cached_answer
            return

        # Step 2: 意图识别
        intent = self.intent_recognizer.recognize(query)

        if intent == 'general':
            logger.info(f"[rag_system.RAGSystem.generate_answer_withhistory:{inspect.currentframe().f_lineno}] 流式生成 - 意图: general")
            prompt_input = self.rag_prompt.format(
                question=query,
                context='',
                phone=Config.CUSTOMER_SERVICE_PHONE,
            )
            if history_context:
                prompt_input = f"以下是之前的对话记录:\n{history_context}\n\n现在请回答新问题:\n{prompt_input}"

            answer_chunks = []
            try:
                for chunk in self.llm.stream(prompt_input):
                    if chunk.content:
                        answer_chunks.append(chunk.content)
                        yield chunk.content
            except Exception:
                logger.exception(f"[rag_system.RAGSystem.generate_answer_withhistory:{inspect.currentframe().f_lineno}] 流式生成-general异常，问题：{query}")
                yield f'抱歉, 处理您的问题时出错, 请联系人工客服: {Config.CUSTOMER_SERVICE_PHONE}'
                return

            full_answer = ''.join(answer_chunks)
            processing_time = time.time() - start_time
            self._record_qa(user_id, session_id, query, full_answer, 'LLM', '直接回答', 0, processing_time)
            self._update_hot_questions(query, full_answer)
            return

        # professional：先检索文档
        logger.info(f"[rag_system.RAGSystem.generate_answer_withhistory:{inspect.currentframe().f_lineno}] 流式生成 - 意图: professional")
        context_docs, strategy = self._retrieve_entry(query, source_filter)
        retrieval_count = len(context_docs) if context_docs else 0

        # 提取来源信息（文件名+页码），去重
        sources_info = []
        seen_sources = set()
        if context_docs:
            for doc in context_docs:
                meta = doc.metadata if hasattr(doc, 'metadata') else {}
                file_name = meta.get('file_name', '未知文件')
                page_start = meta.get('page_start', 0)
                source_key = f"{file_name}_{page_start}"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    sources_info.append({
                        "file_name": file_name,
                        "page": page_start
                    })
            context_list = [doc.page_content for doc in context_docs]
            context = '\n\n'.join(context_list)
        else:
            context = ''

        prompt = self.rag_prompt.format(
            question=query,
            context=context,
            phone=Config.CUSTOMER_SERVICE_PHONE,
        )
        if history_context:
            prompt = f"以下是之前的对话记录:\n{history_context}\n\n现在请回答新问题:\n{prompt}"

        answer_chunks = []
        try:
            for chunk in self.llm.stream(prompt):
                if chunk.content:
                    answer_chunks.append(chunk.content)
                    yield chunk.content
        except Exception:
            logger.exception(f"[rag_system.RAGSystem.generate_answer_withhistory:{inspect.currentframe().f_lineno}] 流式生成-professional异常，问题：{query}")
            yield f'抱歉, 处理您的问题时出错, 请联系人工客服: {Config.CUSTOMER_SERVICE_PHONE}'
            return

        full_answer = ''.join(answer_chunks)
        processing_time = time.time() - start_time
        self._record_qa(user_id, session_id, query, full_answer, 'RAG', strategy, retrieval_count, processing_time)
        self._update_hot_questions(query, full_answer)

        # 将来源信息存储到实例属性，供WebSocket发送
        self._last_sources = sources_info


if __name__ == '__main__':
    rag_system = RAGSystem()

    # rag_system._test_RETRIEVE_DIRECTLY()  # 直接检索
    # rag_system._test_RETRIEVE_BACKTRACKING() # 回溯检索
    # rag_system._test_RETRIEVE_SUBQUERIES() # 子查询检索
    # rag_system._test_RETRIEVE_HYDE() # 测试HyDE检索
    # rag_system._test_RETRIEVE_ENTRY() # 测试检索统一入口

    rag_system._test_GENERATE_ANSWER() # 测试RAG核心流程
