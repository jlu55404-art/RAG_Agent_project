"""导入管理学高频问答CSV到MySQL + 清Redis缓存"""
import csv
import pymysql
import redis
from base import Config

def import_faq():
    # 连接MySQL
    conn = pymysql.connect(
        host=Config.MYSQL_HOST,
        port=Config.MYSQL_PORT,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        database=Config.MYSQL_DB,
        charset='utf8mb4'
    )
    cursor = conn.cursor()

    # 读取CSV
    csv_path = r'D:\RAG_project\mysql_qa\data\管理学高频问答.csv'
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row['question'], row['answer'], row.get('category', '')))

    print(f"CSV共读取 {len(rows)} 条问答")

    # 清空旧数据
    cursor.execute("DELETE FROM management_faq")
    print(f"已清空management_faq表")

    # 插入新数据
    for q, a, c in rows:
        cursor.execute(
            "INSERT INTO management_faq (question, answer, category) VALUES (%s, %s, %s)",
            (q, a, c)
        )
    conn.commit()
    print(f"已插入 {len(rows)} 条问答到management_faq表")

    cursor.close()
    conn.close()

    # 清Redis缓存
    r = redis.Redis(
        host=Config.REDIS_HOST,
        port=Config.REDIS_PORT,
        password=Config.REDIS_PASSWORD,
        db=Config.REDIS_DB
    )
    r.delete('qa_original_questions')
    r.delete('qa_tokenized_questions')
    print("已清除Redis BM25缓存（qa_original_questions, qa_tokenized_questions）")

    # 也清除RAG答案缓存（旧错误答案）
    keys = r.keys('answer:*')
    if keys:
        r.delete(*keys)
        print(f"已清除 {len(keys)} 条RAG答案缓存")
    else:
        print("无RAG答案缓存需清除")

    print("导入完成！请重启服务使新数据生效。")

if __name__ == '__main__':
    import_faq()
