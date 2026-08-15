from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from base import logger, Config


class StrategySelector:
    """管理学专升本RAG - 检索策略选择器
    根据用户查询，用大模型判断应该用什么检索策略。
    支持四种策略：直接检索、假设问题检索(HyDE)、子查询检索、回溯问题检索。
    """

    def __init__(self):
        self.llm = ChatTongyi(
            api_key=Config.DASHSCOPE_API_KEY,
            model=Config.LLM_MODEL,
            temperature=0.1,
        )
        self.prompt_template = self._get_strategy_prompt()

    def select_strategy(self, query):
        """选择检索策略的核心方法
        :param query: 用户输入的查询文本
        :return: 字符串, 选中的检索策略名称
        """
        prompt = self.prompt_template.format(query=query)

        try:
            response = self.llm.invoke(prompt)
            logger.info(f'query: {query} \nLLM Response: {response}')
            return response.content
        except Exception:
            logger.exception(f'llm调用失败, query: {query}')
            return '直接检索'

    def _get_strategy_prompt(self):
        """定义提示模板，告诉大模型如何根据管理学相关问题选择检索策略"""
        template_str = """# Role
你是一个专业的 RAG（检索增强生成）查询意图分析专家，专注于管理学专升本考试领域。你的唯一任务是：精准识别用户查询的核心意图，并将其映射到最合适的检索策略。

# Strategies & Decision Boundaries
请严格基于以下定义进行判断，不要自行创造策略：

1. **直接检索**
   - 核心特征：【事实明确】、【实体具体】、【无需推理】。
   - 适用：询问管理学具体概念定义、人名理论名称、管理原则内容、具体知识点。
   - 示例："泰勒科学管理理论的主要内容？" / "马斯洛需求层次理论有哪几个层次？" / "法约尔的14条管理原则是什么？" / "管理的四大职能是什么？"

2. **假设问题检索 (HyDE)**
   - 核心特征：【语义抽象】、【概念宽泛】、【需要发散/总结】。
   - 适用：询问"管理学学习方法"、"如何提高管理效率"、"管理理论的发展趋势"等没有标准答案的开放性、论述性问题。
   - 示例："论述管理理论的发展历程" / "如何提高组织的凝聚力？" / "管理学在现代社会中的应用前景？" / "管理对企业发展的重要性？"

3. **子查询检索**
   - 核心特征：【多实体对比】、【多维度并列】、【显式连接词】。
   - 适用：包含"A和B的区别"、"X与Y的比较"、"分别阐述..."等需要拆分检索再合并的管理学对比问题。
   - 示例："比较泰勒科学管理和法约尔行政管理理论的异同" / "领导和管理有什么区别？" / "集权和分权各有什么优缺点？" / "X理论和Y理论的区别是什么？"

4. **回溯问题检索**
   - 核心特征：【场景复杂】、【口语化/冗余】、【隐含前提】、【长难句】。
   - 适用：描述一个具体场景并询问管理学知识，或包含大量口语化描述的复杂问题。
   - 示例："我们公司员工积极性不高，老板应该怎么运用激励理论来调动大家的工作热情？" -> "管理学中激励理论的主要内容和应用方法"
   - 示例："那个什么理论说人都是经济人，是谁提出来的啊？" -> "经济人假设的提出者及理论内容"

# Workflow
1. 分析用户查询 `{query}` 的语义结构。
2. 逐一匹配上述 4 种策略的【核心特征】。
3. 确定唯一最佳策略。

# Output Rules (CRITICAL)
- 禁止输出任何分析过程、推理步骤、标点符号或额外文本。
- 禁止使用 Markdown 格式（如加粗、列表）。
- 仅允许返回以下四个字符串之一：
  直接检索
  假设问题检索
  子查询检索
  回溯问题检索

# Input
Query: {query}

# Output
        """
        return PromptTemplate(
            template=template_str,
            input_variables=["query"],
        )

    def _test_SELECT_STRATEGY(self):
        print('测试 -> 选择检索策略功能：')
        retrieval_strategies = {
            "直接检索": [
                "泰勒科学管理理论的主要内容有哪些？",
                "马斯洛需求层次理论的五个层次分别是什么？",
                "法约尔提出的管理五要素是什么？",
                "管理的四大职能分别是什么？",
                "决策的类型有哪些？",
                "组织结构的常见形式有哪些？",
                "什么是管理的科学性和艺术性？",
                "领导的影响力来源有哪些？",
                "控制的过程包括哪些步骤？",
                "目标管理的特点是什么？"
            ],
            "假设问题检索": [
                "论述管理理论的发展历程",
                "如何提高组织的凝聚力？",
                "管理学在现代社会中有哪些应用？",
                "管理对企业发展的重要性体现在哪些方面？",
                "怎样做好一名优秀的管理者？",
                "企业文化对管理有什么影响？",
                "如何运用激励理论提高员工积极性？",
                "管理学的基本原理在实践中如何运用？",
                "现代管理理论有哪些发展趋势？",
                "如何建立良好的组织沟通机制？"
            ],
            "子查询检索": [
                "比较泰勒科学管理和法约尔行政管理理论的异同",
                "领导和管理有什么区别？",
                "集权和分权各有什么优缺点？",
                "X理论和Y理论的区别是什么？",
                "正式组织和非正式组织有什么不同？",
                "前馈控制和反馈控制有什么区别？",
                "战略管理和战术管理的区别是什么？",
                "民主式领导和专制式领导各有什么特点？",
                "机械式组织和有机式组织有何不同？",
                "程序化决策和非程序化决策有什么区别？"
            ],
            "回溯问题检索": [
                "我们公司最近员工士气低落，应该用什么管理理论来激励大家？",
                "那个谁提出来说人天生就是懒惰的，不喜欢工作？",
                "考试老是考那个双因素理论，到底是哪两个因素啊？",
                "我总觉得管理学里面那个什么曲线，说的是管理幅度和管理层级的关系，叫什么来着？",
                "我们部门人太多了沟通效率很低，是不是组织结构有问题？",
                "老板想控制成本又不想降低质量，管理学里有没有什么方法？",
                "那个把管理分成计划组织领导控制的理论是谁提出来的？",
                "考试会不会考到管理方格理论？那个理论讲什么的？",
                "我们公司制度太死了没有创新，管理学怎么说的？",
                "为什么说管理既是科学又是艺术？"
            ]
        }

        import random
        random_category = random.choice(list(retrieval_strategies.keys()))
        random_query = random.choice(retrieval_strategies[random_category])
        print(f"【随机问题】: {random_query}")
        print(f"【所属分类】: {random_category}")
        ret = self.select_strategy(query=random_query)
        print(f"【模型输出】: {ret}")


if __name__ == '__main__':
    ss = StrategySelector()
    ss._test_SELECT_STRATEGY()
