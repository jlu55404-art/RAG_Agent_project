# -*- coding: utf-8 -*-
"""
数据质量检查脚本 - 检测FAQ题库和向量库的过拟合问题
"""
import csv
import re
import json
from collections import Counter, defaultdict

def load_faq(csv_path):
    """加载FAQ数据"""
    data = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def jieba_tokenize(text):
    """简单分词（不依赖jieba，用正则做字符级和词级分析）"""
    # 去除标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    # 返回字符列表（用于BM25模拟）
    return list(text)

def calc_char_overlap(q1, q2):
    """计算两个问题的字符级重叠率"""
    c1 = set(re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', q1))
    c2 = set(re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', q2))
    if not c1 or not c2:
        return 0
    intersection = c1 & c2
    union = c1 | c2
    return len(intersection) / len(union)  # Jaccard相似度

def calc_word_overlap(q1, q2):
    """计算两个问题的词级重叠率（用2-gram模拟）"""
    t1 = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', q1)
    t2 = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', q2)
    # 2-gram
    ngrams1 = set(t1[i:i+2] for i in range(len(t1)-1)) if len(t1) > 1 else {t1}
    ngrams2 = set(t2[i:i+2] for i in range(len(t2)-1)) if len(t2) > 1 else {t2}
    if not ngrams1 or not ngrams2:
        return 0
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    return len(intersection) / len(union)

def calc_answer_overlap(a1, a2):
    """计算两个答案的内容重叠率"""
    # 取前200个字符比较（避免长文本计算太慢）
    t1 = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', a1[:300])
    t2 = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', a2[:300])
    ngrams1 = set(t1[i:i+3] for i in range(len(t1)-2)) if len(t1) > 2 else {t1}
    ngrams2 = set(t2[i:i+3] for i in range(len(t2)-2)) if len(t2) > 2 else {t2}
    if not ngrams1 or not ngrams2:
        return 0
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    return len(intersection) / len(union)

def check_faq_quality(data):
    """检查FAQ题库质量"""
    print("=" * 70)
    print("  FAQ题库数据质量检查报告")
    print("=" * 70)

    total = len(data)
    print(f"\n【基本统计】")
    print(f"  FAQ总数: {total} 条")

    # 1. 分类统计
    categories = Counter(row.get('category', '未知') for row in data)
    print(f"\n【分类分布】")
    for cat, count in categories.most_common():
        bar = "█" * count
        print(f"  {cat}: {count}条 {bar}")

    # 2. 答案长度统计
    answer_lengths = [len(row['answer']) for row in data]
    avg_len = sum(answer_lengths) / len(answer_lengths)
    print(f"\n【答案长度统计】")
    print(f"  平均长度: {avg_len:.0f} 字符")
    print(f"  最短答案: {min(answer_lengths)} 字符")
    print(f"  最长答案: {max(answer_lengths)} 字符")

    # 短答案（可能信息量不足）
    short_answers = [(i+1, row['question'][:30], len(row['answer']))
                     for i, row in enumerate(data) if len(row['answer']) < 100]
    if short_answers:
        print(f"\n  ⚠️ 过短答案（<100字符，信息量可能不足）:")
        for idx, q, l in short_answers:
            print(f"     #{idx} [{l}字符] {q}...")

    # 3. 问题相似度检查（核心：检测过拟合）
    print(f"\n{'='*70}")
    print(f"【问题相似度检查 - 检测BM25误匹配风险】")
    print(f"{'='*70}")

    high_sim_pairs = []  # 高相似度问题对
    medium_sim_pairs = []  # 中相似度问题对

    for i in range(total):
        for j in range(i+1, total):
            q1 = data[i]['question']
            q2 = data[j]['question']
            char_sim = calc_char_overlap(q1, q2)
            word_sim = calc_word_overlap(q1, q2)
            # 综合相似度
            combined_sim = (char_sim + word_sim) / 2

            if combined_sim > 0.6:
                high_sim_pairs.append((i+1, j+1, q1, q2, combined_sim, char_sim, word_sim))
            elif combined_sim > 0.4:
                medium_sim_pairs.append((i+1, j+1, q1, q2, combined_sim, char_sim, word_sim))

    # 高相似度对（>0.6）—— BM25极有可能误匹配
    if high_sim_pairs:
        print(f"\n  🔴 高相似度问题对（综合相似度 > 0.6，BM25极可能误匹配）: {len(high_sim_pairs)}对")
        for idx1, idx2, q1, q2, sim, csim, wsim in sorted(high_sim_pairs, key=lambda x: -x[4]):
            print(f"\n    FAQ#{idx1} vs FAQ#{idx2} [相似度: {sim:.2f}]")
            print(f"      Q1: {q1[:50]}")
            print(f"      Q2: {q2[:50]}")
            # 检查是否是同一个问题的变体
            a1 = data[idx1-1]['answer']
            a2 = data[idx2-1]['answer']
            ans_sim = calc_answer_overlap(a1, a2)
            if ans_sim > 0.5:
                print(f"      ⚠️ 答案也高度相似（答案相似度: {ans_sim:.2f}）→ 建议合并！")
            else:
                print(f"      ✅ 答案不同（答案相似度: {ans_sim:.2f}）→ 问题相似但考不同知识点")
    else:
        print(f"\n  ✅ 没有高相似度问题对（>0.6）")

    # 中相似度对（0.4-0.6）
    if medium_sim_pairs:
        print(f"\n  🟡 中相似度问题对（综合相似度 0.4-0.6，BM25可能误匹配）: {len(medium_sim_pairs)}对")
        for idx1, idx2, q1, q2, sim, csim, wsim in sorted(medium_sim_pairs, key=lambda x: -x[4])[:10]:
            print(f"\n    FAQ#{idx1} vs FAQ#{idx2} [相似度: {sim:.2f}]")
            print(f"      Q1: {q1[:50]}")
            print(f"      Q2: {q2[:50]}")

    # 4. 答案重复检查
    print(f"\n{'='*70}")
    print(f"【答案重复检查 - 检测答案过拟合】")
    print(f"{'='*70}")

    answer_dup_pairs = []
    for i in range(total):
        for j in range(i+1, total):
            ans_sim = calc_answer_overlap(data[i]['answer'], data[j]['answer'])
            if ans_sim > 0.6:
                answer_dup_pairs.append((i+1, j+1, data[i]['question'], data[j]['question'], ans_sim))

    if answer_dup_pairs:
        print(f"\n  🔴 高相似度答案对（答案相似度 > 0.6）: {len(answer_dup_pairs)}对")
        for idx1, idx2, q1, q2, sim in sorted(answer_dup_pairs, key=lambda x: -x[4]):
            print(f"\n    FAQ#{idx1} vs FAQ#{idx2} [答案相似度: {sim:.2f}]")
            print(f"      Q1: {q1[:40]}")
            print(f"      Q2: {q2[:40]}")
            print(f"      ⚠️ 两个不同问题返回几乎相同的答案 → 可能答案过于笼统")
    else:
        print(f"\n  ✅ 没有高相似度答案对")

    # 5. 关键词覆盖检查
    print(f"\n{'='*70}")
    print(f"【关键词覆盖检查 - 检测知识点遗漏】")
    print(f"{'='*70}")

    key_persons = [
        ("泰勒", "Taylor"),
        ("法约尔", "Fayol"),
        ("韦伯", "Weber"),
        ("梅奥", "Mayo"),
        ("马斯洛", "Maslow"),
        ("赫茨伯格", "Herzberg"),
        ("麦格雷戈", "McGregor"),
        ("西蒙", "Simon"),
        ("德鲁克", "Drucker"),
        ("勒温", "Lewin"),
        ("布莱克", "Blake"),
        ("戴明", "Deming"),
    ]

    key_concepts = [
        "科学管理", "一般管理", "官僚组织", "霍桑实验", "人际关系",
        "需求层次", "双因素", "X理论", "Y理论", "决策理论",
        "目标管理", "管理方格", "计划", "组织", "领导", "控制",
        "战略", "变革", "激励", "沟通", "组织文化", "决策",
        "管理幅度", "管理层级", "全面质量管理",
    ]

    all_questions_text = " ".join(row['question'] for row in data)
    all_answers_text = " ".join(row['answer'] for row in data)

    print(f"\n  管理学家覆盖:")
    for cn, en in key_persons:
        q_count = all_questions_text.count(cn)
        a_count = all_answers_text.count(cn)
        status = "✅" if q_count > 0 or a_count > 0 else "❌ 未覆盖"
        print(f"    {cn}({en}): 问题{q_count}次, 答案{a_count}次 {status}")

    print(f"\n  核心概念覆盖:")
    uncovered = []
    for concept in key_concepts:
        q_count = all_questions_text.count(concept)
        a_count = all_answers_text.count(concept)
        total_count = q_count + a_count
        status = "✅" if total_count > 0 else "❌ 未覆盖"
        if total_count == 0:
            uncovered.append(concept)
        print(f"    {concept}: 问题{q_count}次, 答案{a_count}次 {status}")

    if uncovered:
        print(f"\n  ⚠️ 未覆盖的核心概念: {', '.join(uncovered)}")
        print(f"     建议: 为这些概念添加FAQ条目")

    # 6. 综合评估
    print(f"\n{'='*70}")
    print(f"【综合评估】")
    print(f"{'='*70}")

    issues = []
    if high_sim_pairs:
        issues.append(f"高相似问题对: {len(high_sim_pairs)}对（BM25误匹配风险高）")
    if answer_dup_pairs:
        issues.append(f"高相似答案对: {len(answer_dup_pairs)}对（答案可能过于笼统）")
    if uncovered:
        issues.append(f"未覆盖核心概念: {len(uncovered)}个")
    if short_answers:
        issues.append(f"过短答案: {len(short_answers)}条")

    # 检查分类均衡性
    cat_counts = list(categories.values())
    if max(cat_counts) > min(cat_counts) * 3:
        issues.append(f"分类不均衡: 最大{max(cat_counts)}条 vs 最小{min(cat_counts)}条")

    if issues:
        print(f"\n  ⚠️ 发现 {len(issues)} 个潜在问题:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print(f"\n  ✅ 数据质量良好，未发现明显过拟合问题")

    # 7. 过拟合风险评估
    print(f"\n{'='*70}")
    print(f"【过拟合风险评估】")
    print(f"{'='*70}")

    overfit_score = 0
    if high_sim_pairs:
        overfit_score += len(high_sim_pairs) * 2
    if answer_dup_pairs:
        overfit_score += len(answer_dup_pairs) * 2
    if len(medium_sim_pairs) > 5:
        overfit_score += len(medium_sim_pairs)

    if overfit_score == 0:
        risk = "低"
        emoji = "🟢"
    elif overfit_score <= 5:
        risk = "中"
        emoji = "🟡"
    else:
        risk = "高"
        emoji = "🔴"

    print(f"\n  {emoji} 过拟合风险等级: {risk} (得分: {overfit_score})")
    print(f"\n  说明:")
    print(f"  - '过拟合'在FAQ场景下指: 多条FAQ问的是同一件事,")
    print(f"    导致BM25匹配时可能命中错误的FAQ条目")
    print(f"  - 高相似问题对越多, BM25误匹配的概率越大")
    print(f"  - 答案重复意味着不同问题给出相同回答, 降低用户体验")

    return issues


def check_milvus_quality():
    """检查Milvus向量库数据质量"""
    print(f"\n\n{'#'*70}")
    print(f"  Milvus向量库数据质量检查")
    print(f"{'#'*70}")

    try:
        from pymilvus import connections, Collection
        from base.config import Config

        connections.connect(
            host=Config.MILVUS_HOST,
            port=Config.MILVUS_PORT,
        )

        collection = Collection("management_qa")
        collection.load()

        # 基本统计
        num_entities = collection.num_entities
        print(f"\n  总向量数: {num_entities}")

        # 查询所有数据检查重复
        results = collection.query(
            expr="id != ''",
            output_fields=["id", "text", "file_name", "page_start", "page_end", "parent_id", "source", "version"]
        )

        print(f"  查询到文档数: {len(results)}")

        # 1. 文件分布
        file_counter = Counter(r['file_name'] for r in results)
        print(f"\n  【文件分布】")
        for f, c in file_counter.most_common():
            print(f"    {f}: {c}个文档块")

        # 2. 重复ID检查（理论上不应该有，因为用MD5做主键）
        id_counter = Counter(r['id'] for r in results)
        dup_ids = {k: v for k, v in id_counter.items() if v > 1}
        if dup_ids:
            print(f"\n  🔴 重复ID: {len(dup_ids)}个")
            for id_val, count in list(dup_ids.items())[:5]:
                print(f"    {id_val}: 出现{count}次")
        else:
            print(f"\n  ✅ 无重复ID（MD5主键去重有效）")

        # 3. 空文本检查
        empty_texts = [r for r in results if not r.get('text', '').strip()]
        if empty_texts:
            print(f"\n  🔴 空文本: {len(empty_texts)}个文档块文本为空")
        else:
            print(f"  ✅ 无空文本")

        # 4. 页码检查
        no_page = [r for r in results if r.get('page_start', 0) == 0]
        if no_page:
            print(f"\n  ⚠️ 无页码信息: {len(no_page)}个文档块（引用标注会缺失页码）")
        else:
            print(f"  ✅ 所有文档块都有页码信息")

        # 5. 父块关联检查
        no_parent = [r for r in results if not r.get('parent_id', '').strip()]
        if no_parent:
            print(f"\n  ⚠️ 无父块关联: {len(no_parent)}个文档块（RAG检索时可能丢失上下文）")
        else:
            print(f"  ✅ 所有文档块都有父块关联")

        # 6. 文本长度分布
        text_lengths = [len(r.get('text', '')) for r in results]
        avg_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        short_blocks = [r for r in results if len(r.get('text', '')) < 50]
        long_blocks = [r for r in results if len(r.get('text', '')) > 500]
        print(f"\n  【文本长度分布】")
        print(f"    平均长度: {avg_len:.0f} 字符")
        print(f"    最短: {min(text_lengths)} 字符")
        print(f"    最长: {max(text_lengths)} 字符")
        if short_blocks:
            print(f"    ⚠️ 过短块(<50字): {len(short_blocks)}个（可能信息量不足）")
        if long_blocks:
            print(f"    ⚠️ 过长块(>500字): {len(long_blocks)}个（可能切分粒度太大）")

        # 7. 父块去重检查（同一个parent_id有多少子块）
        parent_counter = Counter(r.get('parent_id', '') for r in results)
        print(f"\n  【父块-子块映射】")
        print(f"    独立父块数: {len(parent_counter)}")
        print(f"    平均每个父块含子块: {len(results)/len(parent_counter):.1f}个")
        # 检查是否有过多子块属于同一父块
        large_parents = {k: v for k, v in parent_counter.items() if v > 10}
        if large_parents:
            print(f"    ⚠️ 子块过多的父块(>10): {len(large_parents)}个")
            for pid, cnt in sorted(large_parents.items(), key=lambda x: -x[1])[:5]:
                print(f"      {pid}: {cnt}个子块")

        # 8. 重复内容检查（不同ID但内容相同）
        text_counter = Counter(r.get('text', '')[:100] for r in results)
        dup_texts = {k: v for k, v in text_counter.items() if v > 1 and len(k) > 20}
        if dup_texts:
            print(f"\n  🟡 相似内容块（前100字相同）: {len(dup_texts)}组")
            print(f"    说明: 可能有重复内容被多次向量化，浪费存储和检索资源")
            for text_prefix, count in sorted(dup_texts.items(), key=lambda x: -x[1])[:5]:
                print(f"      [{count}次] {text_prefix[:60]}...")
        else:
            print(f"\n  ✅ 无明显重复内容")

        connections.disconnect("default")

    except Exception as e:
        print(f"\n  ❌ 连接Milvus失败: {e}")
        print(f"     请确认Milvus服务正在运行")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"D:\RAG_project")

    # 1. FAQ题库检查
    csv_path = r"D:\RAG_project\mysql_qa\data\管理学高频问答.csv"
    faq_data = load_faq(csv_path)
    issues = check_faq_quality(faq_data)

    # 2. Milvus向量库检查
    check_milvus_quality()

    print(f"\n\n{'='*70}")
    print(f"  检查完成！")
    print(f"{'='*70}")
