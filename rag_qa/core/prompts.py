from langchain_core.prompts import PromptTemplate


class RAGPrompts:
    """管理学专升本RAG问答系统 - 全部Prompt模板集中管理

    包含4种检索策略对应的提示词模板：
    1. 直接检索 rag_prompt         → 搜到文档后直接生成答案
    2. 先猜后搜 hyde_prompt        → 先让大模型猜一个答案，用猜的答案去搜
    3. 子查询检索 subquery_prompt   → 把复杂问题拆成几个简单问题分别搜
    4. 回溯检索 backtracking_prompt → 把啰嗦的问题简化成核心问题再搜
    """

    # ===== 1. 基础RAG提示词：管理学专升本场景定制 =====
    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""你是"EduRAG管理学(河南)"的智能答疑助手，专门帮助河南专升本考生复习管理学知识。

请遵循以下规则回答问题：
1. 基于提供的上下文（教材内容）回答，结合上下文中的相关信息进行归纳总结
2. 即使上下文没有直接的标准答案，也要根据相关内容尽力回答，帮助考生理解知识点
3. 涉及人名、理论名称时，请完整写出全名
4. 简答题请按要点分条作答，条理清晰
5. 适当标注"考试重点"、"高频考点"等提示，帮助考生备考
6. 如果涉及对比题，请用表格形式呈现
7. 只有当上下文与问题完全无关、无法提供任何有价值的信息时，才回复："该问题超出当前知识库范围，建议查阅管理学教材或联系EduRAG管理员，电话：{phone}。"

上下文: {context}
问题: {question}
回答:""",
            input_variables=["context", "question", "phone"],
        )

    # ===== 2. HyDE提示词：先猜答案再搜索（管理学场景） =====
    @staticmethod
    def hyde_prompt():
        return PromptTemplate(
            template="""假设你是一位管理学教授，正在编写河南专升本管理学考试教材。请针对以下问题，生成一段专业的假设性答案（包含管理学专业术语和理论名称）：

问题: {query}
假设答案:""",
            input_variables=["query"],
        )

    # ===== 3. 子查询提示词：复杂问题拆简单（管理学场景） =====
    @staticmethod
    def subquery_prompt():
        return PromptTemplate(
            template="""将以下管理学相关的复杂查询分解为多个简单的子查询，每行一个子查询。
每个子查询应该能独立检索到相关的管理学知识点。

查询: {query}
子查询:""",
            input_variables=["query"],
        )

    # ===== 4. 回溯提示词：简化啰嗦问题（管理学场景） =====
    @staticmethod
    def backtracking_prompt():
        return PromptTemplate(
            template="""将以下关于管理学的复杂/口语化查询简化为一个精准的学术问题，保留核心管理学概念：

查询: {query}
简化问题:""",
            input_variables=["query"],
        )


if __name__ == '__main__':
    rag_prompt = RAGPrompts.rag_prompt()
    result = rag_prompt.format(
        context="泰勒的科学管理理论主要包括：1.工作定额原理 2.标准化原理 3.能力与工作相适应原理 "
                "4.差别计件付酬制 5.计划职能同执行职能相分离",
        question="泰勒科学管理理论的主要内容有哪些？",
        phone="***********"
    )
    print(f'基础RAG模板结果:\n{result}')
