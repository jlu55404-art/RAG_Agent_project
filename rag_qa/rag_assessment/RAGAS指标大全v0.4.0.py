
# TODO ctrl+左键 ↓ 点击查看ragas的全部metrics指标
# todo 这个文件是一个"指标速查表"：把 ragas 库中所有评估指标按分类整理出来
# todo 不需要运行，只是用来浏览有哪些指标可用、每个指标是干什么的
import ragas.metrics
# ==============================================
# 一、核心评估指标（单轮对话、最常用）
# ==============================================

# 1. 答案正确性指标（需要标准答案）
# todo AnswerCorrectness：判断生成的答案和标准答案比，是不是"对的"（看事实对不对、信息全不全）
from ragas.metrics._answer_correctness import AnswerCorrectness, answer_correctness
# AnswerCorrectness: 评估生成答案与标准答案的整体正确性（包含事实准确性、完整性等）
# answer_correctness: 对应的可直接使用的指标实例

# 2. 答案相关性指标（评估答案与问题的匹配度）
from ragas.metrics._answer_relevance import (
    AnswerRelevancy,          # 答案相关性：评估答案是否直接、完整地回答了用户问题
    ResponseRelevancy,       # 响应相关性：和AnswerRelevancy类似，侧重对话响应层面的相关性
    answer_relevancy,         # 对应的可直接使用的指标实例
)

# 3. 答案相似度指标（基于语义/字符串的相似度）
from ragas.metrics._answer_similarity import (
    AnswerSimilarity,         # 答案相似度：计算生成答案与标准答案的语义相似度
    SemanticSimilarity,      # 语义相似度：仅基于Embedding计算两段文本的语义相似度
    answer_similarity,        # 对应的可直接使用的指标实例
)

# 4. 忠实度指标（评估模型是否基于上下文生成，无幻觉）
from ragas.metrics._faithfulness import Faithfulness, FaithfulnesswithHHEM, faithfulness
# Faithfulness: 忠实度：判断答案中的每个事实是否都能在上下文找到依据（核心抗幻觉指标）
# FaithfulnesswithHHEM: 基于HHEM模型的忠实度，精度更高，速度更慢
# faithfulness: 对应的可直接使用的指标实例

# 5. 上下文精确率（评估检索结果的排序质量）
from ragas.metrics._context_precision import (
    ContextPrecision,                          # 上下文精确率：按检索顺序评估，越靠前的有效信息权重越高
    ContextUtilization,                        # 上下文利用率：评估答案对上下文信息的利用程度
    IDBasedContextPrecision,                   # 基于ID匹配的上下文精确率（无LLM，速度快）
    LLMContextPrecisionWithoutReference,       # 无参考的LLM上下文精确率（无需标准答案）
    LLMContextPrecisionWithReference,          # 有参考的LLM上下文精确率（需要标准答案）
    NonLLMContextPrecisionWithReference,       # 无LLM的上下文精确率（纯字符串匹配，适合快速评估）
    context_precision,                         # 对应的可直接使用的指标实例
)

# 6. 上下文召回率（评估检索结果的完整性）
from ragas.metrics._context_recall import (
    ContextRecall,                # 上下文召回率：评估检索结果是否包含回答问题所需的所有关键信息
    IDBasedContextRecall,          # 基于ID匹配的上下文召回率（无LLM，速度快）
    LLMContextRecall,              # 基于LLM的上下文召回率（更精准，需要标准答案）
    NonLLMContextRecall,           # 无LLM的上下文召回率（纯字符串匹配，适合快速评估）
    context_recall,                # 对应的可直接使用的指标实例
)

# 7. 上下文相关性（评估检索结果的整体质量）
from ragas.metrics._nv_metrics import (
    AnswerAccuracy,        # 答案准确率：评估答案与标准答案的匹配度
    ContextRelevance,      # 上下文相关性：评估检索到的上下文与问题的整体相关性（你之前找的就是这个）
    ResponseGroundedness,  # 响应接地性：评估答案是否完全基于上下文，无额外信息
)

# 8. 其他文本相似度/匹配度指标
from ragas.metrics._bleu_score import BleuScore          # BLEU分数：机器翻译常用的字符串相似度指标
from ragas.metrics._chrf_score import ChrfScore          # CHRF分数：基于字符n-gram的相似度指标
from ragas.metrics._rouge_score import RougeScore        # ROUGE分数：文本摘要常用的字符串相似度指标
from ragas.metrics._string import (
    DistanceMeasure,        # 距离度量：如编辑距离等字符串距离计算
    ExactMatch,             # 精确匹配：判断生成文本是否和标准答案完全一致
    NonLLMStringSimilarity, # 无LLM字符串相似度：纯基于文本的相似度计算
    StringPresence,         # 字符串存在性：判断关键字符串是否在生成文本中出现
)


# ==============================================
# 二、特殊场景/扩展指标
# ==============================================
# todo 下面这些指标比较"专"：只在特定场景下用（如摘要、SQL、Agent、多模态等）

# 1. 实体召回率（评估检索结果中关键实体的覆盖情况）
from ragas.metrics._context_entities_recall import (
    ContextEntityRecall,    # 上下文实体召回率：评估上下文是否包含标准答案中的关键实体
    context_entity_recall,  # 对应的可直接使用的指标实例
)

# 2. 摘要质量指标
from ragas.metrics._summarization import SummarizationScore, summarization_score
# SummarizationScore: 摘要分数：从相关性、一致性、完整性等多维度评估摘要质量

# 3. 噪声敏感度（评估模型对噪声上下文的鲁棒性）
from ragas.metrics._noise_sensitivity import NoiseSensitivity
# NoiseSensitivity: 噪声敏感度：判断模型是否会被无关的上下文信息干扰生成错误答案

# 4. 事实正确性（比Faithfulness更侧重事实准确性）
from ragas.metrics._factual_correctness import FactualCorrectness
# FactualCorrectness: 事实正确性：评估生成答案中的事实是否全部准确无误

# 5. 目标/意图相关指标
from ragas.metrics._goal_accuracy import (
    AgentGoalAccuracyWithoutReference, # 无参考的智能体目标准确率：评估智能体是否达成用户目标
    AgentGoalAccuracyWithReference,    # 有参考的智能体目标准确率：需要标准答案评估目标达成度
)
from ragas.metrics._topic_adherence import TopicAdherenceScore
# TopicAdherenceScore: 主题贴合度：评估对话是否始终围绕用户问题的主题展开

# 6. SQL相关指标
from ragas.metrics._sql_semantic_equivalence import LLMSQLEquivalence
# LLMSQLEquivalence: SQL语义等价性：评估生成的SQL是否与标准答案的SQL语义等价

# 7. 工具调用相关指标（适合Agent场景）
from ragas.metrics._tool_call_accuracy import ToolCallAccuracy # 工具调用准确率：评估Agent是否调用了正确的工具
from ragas.metrics._tool_call_f1 import ToolCallF1              # 工具调用F1值：综合评估工具调用的准确率和召回率

# 8. 多模态指标（适合图文/视频RAG场景）
from ragas.metrics._multi_modal_faithfulness import (
    MultiModalFaithfulness,    # 多模态忠实度：评估多模态答案是否完全基于上下文（文本+图片/视频）
    multimodal_faithness,      # 对应的可直接使用的指标实例
)
from ragas.metrics._multi_modal_relevance import (
    MultiModalRelevance,       # 多模态相关性：评估多模态答案与用户问题的相关性
    multimodal_relevance,      # 对应的可直接使用的指标实例
)

# 9. 特定领域/自定义指标
from ragas.metrics._aspect_critic import AspectCritic               # 维度批评：从自定义维度（如礼貌性、专业性）评估文本质量
from ragas.metrics._domain_specific_rubrics import RubricsScore     # 领域特定评分：基于自定义评分标准评估文本质量
from ragas.metrics._instance_specific_rubrics import InstanceRubrics # 实例特定评分：为每个样本定义单独的评分标准
from ragas.metrics._simple_criteria import SimpleCriteriaScore      # 简单标准评分：基于自然语言描述的简单标准评估文本
from ragas.metrics._datacompy_score import DataCompyScore            # 数据对比分数：评估生成的结构化数据与标准答案的一致性


# ==============================================
# 三、指标基类/类型定义（用于自定义指标）
# ==============================================
# todo 如果你想自己写一个新的评估指标，就继承这些基类，不用从零开始
from ragas.metrics.base import (
    Metric,                  # 所有指标的基类
    MetricOutputType,         # 指标输出类型（数值/离散值等）
    MetricType,              # 指标类型（单轮/多轮、LLM/非LLM等）
    MetricWithEmbeddings,     # 需要Embedding模型的指标基类
    MetricWithLLM,           # 需要LLM模型的指标基类
    MultiTurnMetric,         # 多轮对话指标基类
    SimpleBaseMetric as BaseMetric, # 简化的指标基类
    SimpleLLMMetric as LLMMetric,   # 简化的LLM指标基类
    SingleTurnMetric,        # 单轮对话指标基类
)

# 指标装饰器/工具
# todo 装饰器是 Python 的一种语法糖，可以在不修改原函数的情况下给它加功能
# todo 这里的装饰器用来标记指标的类型（离散值、数值型、排序类）
from ragas.metrics.discrete import DiscreteMetric, discrete_metric # 离散值指标基类/装饰器
from ragas.metrics.numeric import NumericMetric, numeric_metric   # 数值型指标基类/装饰器
from ragas.metrics.ranking import RankingMetric, ranking_metric  # 排序类指标基类/装饰器
from ragas.metrics.result import MetricResult                   # 指标结果对象，用于存储和展示评估结果


# ==============================================
# 四、导出的所有指标/类型（供外部使用）
# ==============================================
# todo __all__ 是 Python 的一个特殊变量，控制 from xxx import * 时导出哪些名字
# todo 这里把上面定义的所有指标和类都列出来，供其他模块导入使用
__all__ = [
    # ========== 指标基础父类 & 通用类型 ==========
    "Metric",                  # 所有评估指标的基类
    "MetricType",              # 指标类型枚举（LLM/非LLM、排序类等）
    "MetricWithEmbeddings",    # 依赖向量模型的指标基类
    "MetricWithLLM",           # 依赖大模型的指标基类
    "SingleTurnMetric",        # 单轮对话指标基类
    "MultiTurnMetric",         # 多轮对话指标基类
    "MetricOutputType",        # 指标输出类型（数值/离散）

    # ========== 简化基类 & 工具装饰器 ==========
    "BaseMetric",              # 简化版指标基类
    "LLMMetric",               # 简化版LLM指标基类
    "MetricResult",            # 评估结果存储与展示对象
    "DiscreteMetric",          # 离散值指标基类
    "NumericMetric",           # 数值型指标基类
    "RankingMetric",           # 排序类指标基类
    "discrete_metric",         # 离散指标装饰器
    "numeric_metric",          # 数值指标装饰器
    "ranking_metric",          # 排序指标装饰器

    # ========== 答案正确性 & 相似度指标 ==========
    "AnswerAccuracy",          # 答案准确率（对比标准答案）
    "AnswerCorrectness",       # 答案整体正确性评估（综合事实+完整度）
    "answer_correctness",      # 答案正确性 可直接使用的实例
    "Faithfulness",            # 忠实度：检测模型是否产生幻觉
    "faithfulness",            # 忠实度 可直接使用的实例
    "FaithfulnesswithHHEM",    # 基于HHEM增强版忠实度（精度更高）
    "AnswerSimilarity",        # 答案语义相似度（对比标准答案）
    "answer_similarity",       # 答案相似度 可直接使用的实例
    "AspectCritic",            # 多维度自定义评判指标
    "AnswerRelevancy",         # 答案相关性：答案是否紧扣用户问题
    "answer_relevancy",        # 答案相关性 可直接使用的实例

    # ========== 上下文检索类指标 ==========
    "ContextRelevance",        # 上下文相关性：评估检索内容与问题匹配度
    "ContextPrecision",        # 上下文精确率：评估检索排序与有效信息占比
    "context_precision",       # 上下文精确率 可直接使用的实例
    "ContextUtilization",      # 上下文利用率：答案对检索内容的使用程度
    "SimpleCriteriaScore",     # 简易规则评分指标
    "ContextRecall",           # 上下文召回率：评估检索内容是否覆盖全部关键信息
    "context_recall",          # 上下文召回率 可直接使用的实例
    "ContextEntityRecall",     # 上下文实体召回率：关键实体覆盖情况
    "context_entity_recall",   # 上下文实体召回率 可直接使用的实例
    "ResponseGroundedness",    # 响应可信度：答案是否严格基于上下文

    # ========== 摘要 & 噪声鲁棒性指标 ==========
    "SummarizationScore",      # 摘要综合质量评分
    "summarization_score",     # 摘要评分 可直接使用的实例
    "NoiseSensitivity",        # 噪声敏感度：模型对无效上下文的抗干扰能力
    "RubricsScore",            # 领域自定义评分规则指标

    # ========== 细分版上下文精确率（分有无参考、有无LLM） ==========
    "LLMContextPrecisionWithReference",    # 有参考+LLM 上下文精确率
    "LLMContextPrecisionWithoutReference",  # 无参考+LLM 上下文精确率
    "NonLLMContextPrecisionWithReference", # 有参考+纯规则 上下文精确率
    "IDBasedContextPrecision",             # 基于ID匹配的上下文精确率

    # ========== 细分版上下文召回率（分有无参考、有无LLM） ==========
    "LLMContextRecall",        # 基于LLM的上下文召回率
    "NonLLMContextRecall",     # 纯规则上下文召回率
    "IDBasedContextRecall",    # 基于ID匹配的上下文召回率

    # ========== 事实、自定义规则类指标 ==========
    "FactualCorrectness",      # 事实正确性：校验答案事实细节是否准确
    "InstanceRubrics",         # 单样本专属自定义评分规则

    # ========== 字符串 & 传统文本相似度指标 ==========
    "NonLLMStringSimilarity",  # 纯规则字符串相似度（不依赖模型）
    "ExactMatch",              # 完全匹配：文本是否与标准答案一字不差
    "StringPresence",          # 字符串存在性：判断关键词是否出现
    "BleuScore",               # BLEU分数（机器翻译经典指标）
    "ChrfScore",               # CHRF字符级相似度分数
    "RougeScore",              # ROUGE分数（文本摘要经典指标）

    # ========== 结构化数据 & 专项场景指标 ==========
    "DataCompyScore",          # 结构化数据对比评分
    "LLMSQLEquivalence",       # SQL语义等价性（数据库问答场景）

    # ========== 智能体(Agent)目标 & 工具调用指标 ==========
    "AgentGoalAccuracyWithoutReference", # 无参考 智能体目标准确率
    "AgentGoalAccuracyWithReference",    # 有参考 智能体目标准确率
    "ToolCallF1",              # 工具调用F1值（综合精确率+召回率）
    "ToolCallAccuracy",        # 工具调用准确率

    # ========== 扩展相关性、语义、主题类指标 ==========
    "ResponseRelevancy",       # 响应相关性（对话层面匹配度）
    "SemanticSimilarity",      # 通用语义相似度（两段文本向量相似度）
    "DistanceMeasure",         # 字符串距离度量（编辑距离等）
    "TopicAdherenceScore",     # 主题贴合度：判断内容是否偏离提问主题

    # ========== 多模态指标（图文/音视频RAG） ==========
    "MultiModalFaithfulness",  # 多模态忠实度（图文/音视频防幻觉）
    "multimodal_faithness",    # 多模态忠实度 可直接使用的实例
    "MultiModalRelevance",     # 多模态相关性（图文/音视频内容匹配）
    "multimodal_relevance",    # 多模态相关性 可直接使用的实例
]