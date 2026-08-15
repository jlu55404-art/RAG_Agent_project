# 导入依赖库
# todo re：正则表达式库，用来做模式匹配和替换（比如同义词替换中的单词边界匹配）
import re  # 正则表达式，用来做同义词替换
# todo os：操作系统接口，这里主要用来读环境变量中的 API 密钥
import os  # 读取系统环境变量（存放API密钥）
import time  # 延时休眠，防止API调用限流
import json  # 读写JSON数据集文件
import random  # 随机选择、随机概率控制
import traceback  # 异常详细报错信息（排错用）
# todo tqdm：命令行进度条库，在循环里调用 pbar.update(1) 就能显示进度
from tqdm import tqdm  # 命令行进度条，直观看到生成进度
# todo OpenAI 兼容客户端：可以用来调用任何兼容 OpenAI 接口格式的大模型 API（这里用来调通义千问）
from openai import OpenAI  # 兼容OpenAI接口的客户端，用来调用通义千问

# 加载本地.env环境变量文件（这里注释了，密钥直接从系统环境读取）
# from dotenv import load_dotenv
# load_dotenv()

# ===================== 1. 初始化通义千问客户端 =====================
# 阿里通义千问兼容OpenAI调用格式，所以直接用OpenAI SDK
# todo OpenAI() 创建一个 API 客户端，api_key 是密钥，base_url 指向通义千问的兼容接口
client = OpenAI(
    api_key=os.getenv("ALIYUN_KEY"),  # 从系统环境变量读取API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问接口地址
)

# ===================== 2. 同义词词典（规则改写用） =====================
# 键：原词语；值：同义替换词列表
# 作用：把固定模板问句，随机换成不同口语表达，模拟不同用户说话风格
# todo 这个字典就像是"同义词翻译表"，程序随机从列表中选一个替换原词，让问句看起来像不同人问的
synonym_dict = {
    "什么": ["什么", "啥是", "如何解释", "具体是什么", "请解释"],
    "课程": ["课程", "培训课程", "课程安排", "教学内容", "课"],
    "学费": ["学费", "费用", "报名费", "学习费用", "价格", "收费"],
    "大纲": ["大纲", "课程内容", "教学计划", "讲义"],
    "师资": ["师资", "教师团队", "讲师阵容", "师资力量", "老师", "讲师"],
    "培训": ["培训", "辅导", "学习计划", "教育课程"],
    "在哪里": ["在哪里", "位于何处", "设在何地", "在哪"],
    "介绍": ["介绍", "说明", "讲解", "概述", "讲讲", "说说"],
    "请问": ["请问", "能否告知", "是否可以告诉我", "麻烦说下"],
    "原理": ["原理", "基本思想", "工作机制"],
    "写一个": ["写一个", "编写一个", "生成一个", "创建一个"],
    "等于多少": ["等于多少", "是多少", "结果是啥", "得多少"],
    "如何": ["如何", "怎样", "咋样"],
    "需要": ["需要", "要", "得有", "必须具备"],
    "基础": ["基础", "前提", "背景", "基本知识"]
}


# ===================== 3. 同义词随机替换函数 =====================
# todo apply_synonym_variation：把输入文本中的词语随机替换成同义词，让问句更多样化
# todo 比如"课程费用是多少"可能变成"培训课程收费多少"，听起来像不同人在问
def apply_synonym_variation(text, replace_prob=0.5):
    """
    对输入文本做同义词随机替换，模拟不同用户话术
    :param text: 原始问句文本
    :param replace_prob: 替换概率 0~1，这里默认50%概率触发替换
    :return: 替换后的新文本
    """
    # 遍历同义词字典里每一组词汇
    for word, synonyms in synonym_dict.items():
        # 正则：精准匹配完整单词（避免把词语里的片段误替换）
        # todo \b 是正则中的"单词边界"，确保只匹配完整单词而不是词的一部分
        pattern = r"\b" + re.escape(word) + r"\b"

        # 定义替换逻辑
        def repl(match):
            # 随机判断是否执行替换
            # todo random.random() 返回 0~1 之间的随机小数，小于 replace_prob 就替换
            if random.random() < replace_prob:
                return random.choice(synonyms)  # 随机选一个同义词
            return match.group(0)  # 不替换，保留原词

        # 执行全局替换
        # todo re.sub() 对每个匹配到的词都调用 repl 函数，决定替换还是不替换
        text = re.sub(pattern, repl, text)
    return text


# ===================== 4. 规则模板生成 - 通用知识问句 =====================
def generate_generic_query_rule():
    """
    纯规则模板生成【通用知识类】问句（数学、代码、概念、常识）
    先套固定模板，再走同义词替换，生成多样化问句
    :return: 一条生成好的问句
    """
    # 问句模板，{} 是待填充占位符
    templates = [
        "什么{concept}？",
        "{concept}的定义是什么？",
        "请解释{concept}的原理。",
        "如何运用{concept}？",
        "计算{num1}+{num2}等于多少？",
        "写一个{lang}的{func}函数",
        "为什么{thing}是{state}？"
    ]
    # 各个占位符的候选词库
    concepts = ["AI", "Transformer模型", "Python", "递归", "算法复杂度", "数据结构", "机器学习"]
    langs = ["Python", "Java", "C++"]
    funcs = ["排序", "计算", "打印"]
    things = ["太阳", "水", "风"]
    states = ["热的", "流动的", "无形的"]
    nums = list(range(1, 100))  # 1~99 数字，用于数学计算

    # 随机选一个模板
    t = random.choice(templates)
    replacements = {}

    # 根据模板里的占位符，填充对应随机内容
    if "{concept}" in t:
        replacements = {"concept": random.choice(concepts)}
    elif "{num1}" in t:
        replacements = {"num1": random.choice(nums), "num2": random.choice(nums)}
    elif "{lang}" in t:
        replacements = {"lang": random.choice(langs), "func": random.choice(funcs)}
    elif "{thing}" in t:
        replacements = {"thing": random.choice(things), "state": random.choice(states)}

    # 把占位符替换成实际内容，得到基础问句
    query = t.format(**replacements)
    # 再做一轮同义词改写，增加多样性
    return apply_synonym_variation(query)


# ===================== 5. 规则模板生成 - 专业咨询问句 =====================
def generate_professional_query_rule():
    """
    纯规则模板生成【IT培训咨询类】问句（课程、学费、地点等）
    :return: 一条生成好的问句
    """
    # 培训咨询类模板
    templates = [
        "请问{subject}课程的学费是多少？",
        "{subject}的课程大纲是什么？",
        "{subject}培训的学习周期有多长？",
        "请介绍一下{subject}培训的主要项目内容。",
        "请问{subject}培训地点在哪里？"
    ]
    # 培训科目候选
    subjects = ["JAVA", "AI", "测试", "Web前端", "Python", "大数据", "DevOps"]
    # 随机选模板 + 随机填科目
    t = random.choice(templates)
    query = t.format(subject=random.choice(subjects))
    # 同义词替换丰富话术
    return apply_synonym_variation(query)


# ===================== 6. 调用通义千问大模型通用接口 =====================
# todo generate_with_qwen：封装了对通义千问 API 的调用，统一做异常处理和超时控制
def generate_with_qwen(prompt):
    """
    统一调用通义千问生成文本，增加异常捕获和超时
    :param prompt: 给大模型的提示词
    :return: 大模型返回的问句，失败返回 None
    """
    try:
        # 发起请求调用模型
        # todo client.chat.completions.create：OpenAI 标准接口，发送对话请求给大模型
        completion = client.chat.completions.create(
            model="qwen-plus",  # 指定模型版本（plus 性价比高）
            messages=[{"role": "user", "content": prompt}],  # 对话内容
            temperature=0.9,  # 随机性：值越高回答越天马行空，这里用来生成多样问句
            timeout=60  # 60秒超时，批量生成需要更长时间
        )
        # 提取返回结果并去除首尾空格
        # todo completion.choices[0].message.content：从返回结果中取出模型生成的文本内容
        return completion.choices[0].message.content.strip()
    except Exception as e:
        # 捕获所有异常，打印错误信息
        print(f"❌️ Qwen-Plus调用失败: {e}→", traceback.format_exc())
        return None


# ===================== 7. 大模型生成 - 通用知识问句（批量版） =====================
def generate_generic_query_qwen(batch_size=1):
    """
    通过提示词，让通义千问随机生成通用知识类问句
    支持批量生成：如果batch_size>1，一次生成多个问题
    """
    if batch_size == 1:
        # 原有单条生成逻辑
        prompt = """
        **你是一个用户，请生成一个“通用知识”类的查询。**
        
        **生成要求：**
        1. 每个查询必须不同，避免重复
        2. 覆盖以下多个领域：
           - 数学计算（加减乘除、方程、几何）
           - 编程语法（Python、Java、C++的语法、函数、类等）
           - 大模型概念（Transformer、Attention、GPT、BERT等原理）
           - 科学常识（物理、化学、生物、天文现象）
           - 逻辑推理（谜题、脑筋急转弯）
           - 历史地理（历史事件、地理知识）
        3. 问题文本长度控制在10-30字
        4. 不要使用序号，每行只写一个问题
        5. 不要有任何额外说明文字
        
        **参考示例（请自由发挥，不要局限于此）：**
        3+5等于多少？
        写一个Python排序函数
        什么是神经网络？
        太阳为什么是热的？
        Python中如何定义类？
        什么是注意力机制？
        质数是什么？
        如何计算圆的面积？
        
        **请直接返回查询文本，不要有任何多余说明。**
        """
        return generate_with_qwen(prompt)
    else:
        # 批量生成逻辑
        prompt = f"""
        **你是一个用户，请一次性生成{batch_size}个不同的“通用知识”类查询，每个查询占一行。**

        **生成要求：**
        1. 每个查询必须不同，避免重复
        2. 覆盖以下多个领域：
           - 数学计算（加减乘除、方程、几何）
           - 编程语法（Python、Java、C++的语法、函数、类等）
           - 大模型概念（Transformer、Attention、GPT、BERT等原理）
           - 科学常识（物理、化学、生物、天文现象）
           - 逻辑推理（谜题、脑筋急转弯）
           - 历史地理（历史事件、地理知识）
        3. 问题文本长度控制在10-30字
        4. 不要使用序号，每行只写一个问题
        5. 不要有任何额外说明文字
        
        **参考示例（请自由发挥，不要局限于此）：**
        3+5等于多少？
        写一个Python排序函数
        什么是神经网络？
        太阳为什么是热的？
        Python中如何定义类？
        什么是注意力机制？
        质数是什么？
        如何计算圆的面积？

        **请直接返回{batch_size}行查询文本，每行一个，不要有任何多余说明。**
        """
        response = generate_with_qwen(prompt)
        if response:
            # 按行分割，过滤空行
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            # 移除可能的序号（如 "1. "、"- "、"1、" 等）
            questions = [re.sub(r'^[\d]+[\.\、\s]+', '', q) for q in questions]
            questions = [re.sub(r'^[\-\*\•]\s+', '', q) for q in questions]
            return questions[:batch_size]  # 返回问题列表
        return []


# ===================== 8. 大模型生成 - 专业咨询问句（批量版） =====================
def generate_professional_query_qwen(batch_size=1):
    """
    通过提示词，让通义千问随机生成IT培训咨询类问句
    支持批量生成：如果batch_size>1，一次生成多个问题
    """
    if batch_size == 1:
        # 原有单条生成逻辑
        prompt = """
        你是一个用户，生成一个“专业咨询”类的查询，涉及IT教育培训（如课程详情、师资、费用、周期、地点等）。
        示例：
        - “JAVA课程费用多少？”
        - “AI培训有哪些老师？”
        - “测试课程什么时候开课？”
        请生成一个类似的查询，直接返回查询文本，不要多余说明。
        """
        return generate_with_qwen(prompt)
    else:
        # 批量生成逻辑
        prompt = f"""
        你是一个用户，请一次性生成{batch_size}个不同的“专业咨询”类查询，每个查询占一行。
        查询涉及IT教育培训（如课程详情、师资、费用、周期、地点等）。

        示例（仅供参考，请不要局限于这些例子）：
        - “JAVA课程费用多少？”
        - “AI培训有哪些老师？”
        - “测试课程什么时候开课？”
        - “Python课程学多久？”
        - “前端培训有就业保障吗？”

        请直接返回{batch_size}行查询文本，每行一个，不要有任何多余说明。
        """
        response = generate_with_qwen(prompt)
        if response:
            # 按行分割，过滤空行
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            # 移除可能的序号（如 "1. "、"- "、"1、" 等）
            questions = [re.sub(r'^[\d]+[\.\、\s]+', '', q) for q in questions]
            questions = [re.sub(r'^[\-\*\•]\s+', '', q) for q in questions]
            return questions[:batch_size]  # 返回问题列表
        return []


# ===================== 9. 数据集保存函数 =====================
def save_dataset(dataset, filename, stage_name):
    """
    将数据集写入JSON文件
    :param dataset: 列表格式数据集
    :param filename: 保存文件名
    :param stage_name: 阶段名称（打印日志用）
    """
    with open(filename, "w", encoding="utf-8") as f:
        # 写入JSON，不转义中文、带缩进格式化
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"➡️ {stage_name}：已保存 {len(dataset)} 条数据到 {filename}")


# ===================== 10. 主函数：批量生成整套训练数据集 =====================
# todo generate_training_dataset：整个脚本的核心入口，一键生成全套分类训练数据
# todo 流程：规则生成通用知识1500条 → 规则生成专业咨询1500条 → 大模型补充通用知识1500条 → 大模型补充专业咨询1500条 → 合并打乱
def generate_training_dataset(total_samples=6000):
    """
    入口函数：生成总计 6000 条标注数据集
    分配规则：
    - 总共6000条：通用知识3000条 + 专业咨询3000条
    - 每一类内部：规则生成1500条 + 大模型生成1500条
    """
    # 每大类总数：3000条
    num_per_category = total_samples // 2  # todo // 是整除，6000//2=3000
    # 每大类里：规则生成1500，大模型生成1500
    num_rule = num_per_category // 2
    num_qwen = num_per_category - num_rule

    # 存储最终数据列表
    generic_samples = []  # 通用知识数据集
    professional_samples = []  # 专业咨询数据集
    # 集合做去重（集合查询速度快，重复问句不重复入库）
    generic_set = set()  # 通用知识
    professional_set = set()  # 专业咨询

    # -------- 第一步：规则生成 通用知识 1500条 --------
    print("➡️ 生成规则通用知识数据...")
    # tqdm 显示进度条，循环直到凑够数量
    with tqdm(total=num_rule, desc="Rule-based Generic") as pbar:
        while len(generic_samples) < num_rule:
            q = generate_generic_query_rule()
            if q not in generic_set:  # 去重判断
                generic_set.add(q)
                generic_samples.append({"query": q, "label": "通用知识"})
                pbar.update(1)  # 进度条+1
    # 分批保存
    save_dataset(generic_samples, "rule_generic_1500.json", "规则通用知识")

    # -------- 第二步：规则生成 专业咨询 1500条 --------
    print("➡️ 生成规则专业咨询数据...")
    with tqdm(total=num_rule, desc="Rule-based Professional") as pbar:
        while len(professional_samples) < num_rule:
            q = generate_professional_query_rule()
            if q not in professional_set:
                professional_set.add(q)
                professional_samples.append({"query": q, "label": "专业咨询"})
                pbar.update(1)
    save_dataset(professional_samples, "rule_professional_1500.json", "规则专业咨询")

    # -------- 第三步：大模型补充 通用知识 到3000条（批量生成）--------
    print("➡️ 生成Qwen-Plus通用知识数据（批量模式）...")
    # 批量生成1500个问题
    batch_questions = generate_generic_query_qwen(batch_size=num_qwen)
    if batch_questions:
        print(f"  ✅ 大模型返回了 {len(batch_questions)} 个问题")
        with tqdm(total=len(batch_questions), desc="Qwen-based Generic") as pbar:
            for q in batch_questions:
                if q and q not in generic_set:
                    generic_set.add(q)
                    generic_samples.append({"query": q, "label": "通用知识"})
                    pbar.update(1)
                time.sleep(0.05)  # 轻微延时
    else:
        # 如果批量生成失败，降级为单条生成
        print("  ⚠️ 批量生成失败，降级为单条生成模式...")
        with tqdm(total=num_qwen, desc="Qwen-based Generic (fallback)") as pbar:
            while len(generic_samples) < num_per_category:
                q = generate_generic_query_qwen(batch_size=1)
                if q and q not in generic_set:
                    generic_set.add(q)
                    generic_samples.append({"query": q, "label": "通用知识"})
                    pbar.update(1)
                time.sleep(0.5)

    # 如果去重后数量不足，用规则生成补充
    if len(generic_samples) < num_per_category:
        need = num_per_category - len(generic_samples)
        print(f"  🔧 补充生成 {need} 个通用知识问题...")
        with tqdm(total=need, desc="补充通用知识") as pbar:
            while len(generic_samples) < num_per_category:
                q = generate_generic_query_rule()
                if q not in generic_set:
                    generic_set.add(q)
                    generic_samples.append({"query": q, "label": "通用知识"})
                    pbar.update(1)

    save_dataset(generic_samples, "generic_3000.json", "通用知识（规则+Qwen）")

    # -------- 第四步：大模型补充 专业咨询 到3000条（批量生成）--------
    print("➡️ 生成Qwen-Plus专业咨询数据（批量模式）...")
    # 批量生成1500个问题
    batch_questions = generate_professional_query_qwen(batch_size=num_qwen)
    if batch_questions:
        print(f"  ✅ 大模型返回了 {len(batch_questions)} 个问题")
        with tqdm(total=len(batch_questions), desc="Qwen-based Professional") as pbar:
            for q in batch_questions:
                if q and q not in professional_set:
                    professional_set.add(q)
                    professional_samples.append({"query": q, "label": "专业咨询"})
                    pbar.update(1)
                time.sleep(0.05)
    else:
        # 如果批量生成失败，降级为单条生成
        print("  ⚠️ 批量生成失败，降级为单条生成模式...")
        with tqdm(total=num_qwen, desc="Qwen-based Professional (fallback)") as pbar:
            while len(professional_samples) < num_per_category:
                q = generate_professional_query_qwen(batch_size=1)
                if q and q not in professional_set:
                    professional_set.add(q)
                    professional_samples.append({"query": q, "label": "专业咨询"})
                    pbar.update(1)
                time.sleep(0.5)

    # 如果去重后数量不足，用规则生成补充
    if len(professional_samples) < num_per_category:
        need = num_per_category - len(professional_samples)
        print(f"  🔧 补充生成 {need} 个专业咨询问题...")
        with tqdm(total=need, desc="补充专业咨询") as pbar:
            while len(professional_samples) < num_per_category:
                q = generate_professional_query_rule()
                if q not in professional_set:
                    professional_set.add(q)
                    professional_samples.append({"query": q, "label": "专业咨询"})
                    pbar.update(1)

    save_dataset(professional_samples, "professional_3000.json", "专业咨询（规则+Qwen）")

    # -------- 第五步：合并全部数据 + 随机打乱 --------
    dataset = generic_samples + professional_samples
    random.shuffle(dataset)  # 打乱顺序，防止模型顺序偏差
    # 顺序偏差指模型没有学习数据本身的特征规律，反而错误捕捉到数据排列顺序、位置的规律
    #   并依靠位置来做预测、判断，属于典型的过拟合 + 学习跑偏
    final_filename = "training_dataset_hybrid_6000.json"
    save_dataset(dataset, final_filename, "最终数据集")
    return dataset


# ===================== 程序入口：运行主逻辑 =====================
# todo if __name__ == "__main__"：只有当直接运行这个脚本时才执行，被 import 时不执行
if __name__ == "__main__":
    # 开始生成6000条数据集
    # todo 这里 total_samples=60 是测试用的小数字，正式生成记得改成 6000
    dataset = generate_training_dataset(total_samples=60)
    print(f"✅ 成功生成 {len(dataset)} 条训练数据，保存在 training_dataset_hybrid_6000.json 文件中。")
    # 打印前10条样本，用来检查效果
    print("\n📋 前10条样本预览：")
    for item in dataset[:10]:
        print(json.dumps(item, ensure_ascii=False))