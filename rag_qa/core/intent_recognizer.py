"""意图识别模块 - 规则匹配 + LLM兜底

判断用户问题是否属于管理学领域：
1. 规则匹配（优先）：关键词/人名/理论/考试术语命中 → professional
2. LLM兜底：规则未命中时，调用通义千问判断是否与管理学相关
3. 失败降级：LLM调用失败时默认为 professional（宁可多检索，不遗漏）

分类结果：
    "professional" - 管理学专业问题，走RAG检索
    "general"      - 通用/闲聊问题，LLM直接回答或模板回复
"""

from langchain_community.chat_models.tongyi import ChatTongyi
from base import logger, Config


class IntentRecognizer:
    """管理学专升本RAG - 意图识别器

    优先通过规则匹配（关键词、人名、理论、考试术语）快速判断，
    规则未命中时调用LLM兜底，确保不遗漏专业问题。
    """

    # 分类结果常量
    PROFESSIONAL = "professional"
    GENERAL = "general"

    def __init__(self):
        """初始化意图识别器：构建关键词集合 + 创建LLM实例"""
        # 构建管理学关键词集合（合并所有类别，统一用集合做O(1)查找）
        self._keywords = self._build_keywords()
        logger.info(f"意图识别器已初始化，关键词数量: {len(self._keywords)}")

    def _build_keywords(self):
        """构建管理学关键词集合
        :return: set 包含所有管理学相关关键词的集合
        """
        # 管理学核心概念
        concepts = [
            "管理", "组织", "领导", "决策", "计划", "控制",
            "激励", "沟通", "协调", "效率", "战略", "目标",
        ]

        # 管理学人名
        people = [
            "泰勒", "法约尔", "韦伯", "马斯洛", "赫茨伯格",
            "麦格雷戈", "西蒙", "德鲁克", "梅奥", "巴纳德",
        ]

        # 管理学理论
        theories = [
            "科学管理", "行政管理", "人际关系", "需求层次",
            "双因素", "XY理论", "期望理论", "公平理论",
            "目标管理", "全面质量管理",
        ]

        # 考试相关术语
        exam_terms = [
            "真题", "考点", "重点", "简答题", "论述题",
            "名词解释", "选择题",
        ]

        # 合并所有关键词到一个集合
        all_keywords = set(concepts + people + theories + exam_terms)
        return all_keywords

    def _rule_match(self, query):
        """规则匹配：检查用户问题是否包含管理学关键词
        :param query: 用户输入的查询文本
        :return: bool，True表示命中关键词
        """
        for keyword in self._keywords:
            if keyword in query:
                logger.info(f"规则匹配命中: query='{query}', keyword='{keyword}'")
                return True
        return False

    def _llm_classify(self, query):
        """LLM兜底分类：调用通义千问判断问题是否属于管理学领域
        :param query: 用户输入的查询文本
        :return: bool，True表示属于管理学领域，False表示不属于；
                 调用失败时默认返回True（宁可多检索，不遗漏）
        """
        try:
            llm = ChatTongyi(
                api_key=Config.DASHSCOPE_API_KEY,
                model=Config.LLM_MODEL,
                temperature=0.1,
            )
            prompt = (
                f"请判断以下问题是否与管理学学科相关，只回答'是'或'否'。\n"
                f"问题：{query}"
            )
            response = llm.invoke(prompt)
            answer = response.content.strip()
            logger.info(f"LLM意图识别: query='{query}', answer='{answer}'")

            if "是" in answer:
                return True
            return False

        except Exception:
            logger.exception(f"LLM意图识别失败，降级为professional: query='{query}'")
            # 失败降级：默认为"是"（宁可多检索，不遗漏）
            return True

    def recognize(self, query):
        """意图识别主方法：判断用户问题属于专业问题还是通用问题

        处理流程：
        1. 先做规则匹配（关键词检查），命中直接返回 professional
        2. 规则未命中则调用LLM兜底判断
        3. LLM调用失败时降级返回 professional

        :param query: 用户输入的查询文本
        :return: "professional" 或 "general"
        """
        # 第一步：规则匹配（优先）
        if self._rule_match(query):
            logger.info(f"意图识别结果: professional (规则匹配), query='{query}'")
            return self.PROFESSIONAL

        # 第二步：LLM兜底
        is_management = self._llm_classify(query)
        if is_management:
            logger.info(f"意图识别结果: professional (LLM判断), query='{query}'")
            return self.PROFESSIONAL

        logger.info(f"意图识别结果: general, query='{query}'")
        return self.GENERAL


if __name__ == '__main__':
    recognizer = IntentRecognizer()

    test_queries = [
        "泰勒科学管理理论的主要内容是什么？",
        "马斯洛需求层次理论有几个层次？",
        "今天天气怎么样？",
        "管理的四大职能是什么？",
        "帮我写一首诗",
        "双因素理论的两个因素分别是什么？",
        "管理学专升本真题有哪些重点？",
        "Python怎么安装？",
        "如何提高组织的效率？",
        "1+1等于几？",
    ]

    for q in test_queries:
        result = recognizer.recognize(q)
        print(f"问题: {q}  =>  {result}")
