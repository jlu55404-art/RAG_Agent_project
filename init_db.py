"""
EduRAG管理学(河南)在线问答系统 - 数据库初始化脚本
运行方式: python init_db.py
功能:
  1. 创建MySQL数据库 management_qa 及所有表
  2. 创建Milvus集合 management_qa（含完整Schema）
"""
import sys
import pymysql
from pymilvus import MilvusClient, DataType
from base import Config, logger


# ==================== MySQL 初始化 ====================

MYSQL_TABLES = {
    "users": """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL COMMENT '用户名',
            password_hash VARCHAR(200) NOT NULL COMMENT '密码哈希值',
            role VARCHAR(20) DEFAULT 'user' COMMENT '角色: user/admin',
            total_quota INT DEFAULT 0 COMMENT '总可用次数',
            used_count INT DEFAULT 0 COMMENT '已使用次数',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表'
    """,

    "invite_codes": """
        CREATE TABLE IF NOT EXISTS invite_codes (
            id INT AUTO_INCREMENT PRIMARY KEY,
            code VARCHAR(50) UNIQUE NOT NULL COMMENT '邀请码',
            quota INT NOT NULL COMMENT '赠送使用次数',
            is_used BOOLEAN DEFAULT FALSE COMMENT '是否已使用',
            used_by INT DEFAULT NULL COMMENT '使用者user_id',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            used_at TIMESTAMP NULL COMMENT '使用时间'
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='邀请码表'
    """,

    "conversations": """
        CREATE TABLE IF NOT EXISTS conversations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(100) NOT NULL COMMENT '会话ID',
            user_id INT COMMENT '用户ID',
            question TEXT NOT NULL COMMENT '用户问题',
            answer TEXT NOT NULL COMMENT '系统回答',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '时间戳',
            INDEX idx_session (session_id),
            INDEX idx_user (user_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='对话历史表'
    """,

    "qa_records": """
        CREATE TABLE IF NOT EXISTS qa_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT COMMENT '用户ID',
            session_id VARCHAR(100) COMMENT '会话ID',
            question TEXT NOT NULL COMMENT '用户问题',
            answer TEXT NOT NULL COMMENT '系统回答',
            source VARCHAR(20) COMMENT '来源: FAQ/RAG/LLM',
            strategy VARCHAR(20) COMMENT '检索策略',
            retrieval_count INT DEFAULT 0 COMMENT '检索命中数',
            processing_time FLOAT DEFAULT 0 COMMENT '处理耗时(秒)',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            INDEX idx_user (user_id),
            INDEX idx_source (source),
            INDEX idx_created (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='问答记录表(全量日志)'
    """,

    "hot_questions": """
        CREATE TABLE IF NOT EXISTS hot_questions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question TEXT NOT NULL COMMENT '归一化后的问题',
            answer TEXT NOT NULL COMMENT '最佳答案',
            hit_count INT DEFAULT 1 COMMENT '命中次数',
            last_hit TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后命中时间',
            INDEX idx_hit (hit_count DESC)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='高频问答缓存表'
    """,

    "kb_versions": """
        CREATE TABLE IF NOT EXISTS kb_versions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            version VARCHAR(20) NOT NULL COMMENT '版本号 如v1.0',
            subject VARCHAR(50) NOT NULL COMMENT '学科: management/english/math',
            description TEXT COMMENT '版本说明',
            file_count INT DEFAULT 0 COMMENT '文件数',
            chunk_count INT DEFAULT 0 COMMENT '文档块数',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            INDEX idx_subject (subject),
            INDEX idx_version (version)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库版本管理表'
    """,

    "management_faq": """
        CREATE TABLE IF NOT EXISTS management_faq (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question TEXT NOT NULL COMMENT '问题',
            answer TEXT NOT NULL COMMENT '答案',
            category VARCHAR(50) COMMENT '分类',
            hit_count INT DEFAULT 0 COMMENT '命中次数',
            INDEX idx_category (category)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='管理学FAQ题库(BM25快速匹配)'
    """,
}


def init_mysql():
    """初始化MySQL数据库和所有表"""
    logger.info("[init_db.init_mysql] 开始初始化MySQL数据库...")

    # 先连接MySQL（不指定数据库），创建数据库
    conn = pymysql.connect(
        host=Config.MYSQL_HOST,
        port=Config.MYSQL_PORT,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        charset='utf8mb4'
    )
    cursor = conn.cursor()

    db_name = Config.MYSQL_DB
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci")
    logger.info(f"[init_db.init_mysql] 数据库 `{db_name}` 已就绪")
    cursor.close()
    conn.close()

    # 重新连接，指定数据库，创建所有表
    conn = pymysql.connect(
        host=Config.MYSQL_HOST,
        port=Config.MYSQL_PORT,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        database=db_name,
        charset='utf8mb4'
    )
    cursor = conn.cursor()

    for table_name, create_sql in MYSQL_TABLES.items():
        cursor.execute(create_sql)
        logger.info(f"[init_db.init_mysql] 表 `{table_name}` 已就绪")

    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"[init_db.init_mysql] MySQL初始化完成，共创建 {len(MYSQL_TABLES)} 张表")


# ==================== Milvus 初始化 ====================

def init_milvus():
    """初始化Milvus向量数据库集合（pymilvus 3.0.0 MilvusClient API）"""
    logger.info("[init_db.init_milvus] 开始初始化Milvus集合...")

    uri = f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}"

    # 1. 先用default库连接，创建目标数据库
    client = MilvusClient(uri=uri)
    logger.info(f"[init_db.init_milvus] 已连接Milvus {uri}")

    db_name = Config.MILVUS_DATABASE
    existing_dbs = client.list_databases()
    if db_name not in existing_dbs:
        client.create_database(db_name)
        logger.info(f"[init_db.init_milvus] 数据库 `{db_name}` 已创建")
    else:
        logger.info(f"[init_db.init_milvus] 数据库 `{db_name}` 已存在")
    client.close()

    # 2. 连接到目标数据库
    client = MilvusClient(uri=uri, db_name=db_name)
    logger.info(f"[init_db.init_milvus] 已切换到数据库 `{db_name}`")

    collection_name = Config.MILVUS_COLLECTION

    # 3. 如果集合已存在，先删除（全新初始化）
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        logger.info(f"[init_db.init_milvus] 已删除旧集合 `{collection_name}`")

    # 4. 创建Schema
    schema = client.create_schema(enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="page_start", datatype=DataType.INT64)
    schema.add_field(field_name="page_end", datatype=DataType.INT64)
    schema.add_field(field_name="chapter", datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="version", datatype=DataType.VARCHAR, max_length=20)
    schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)
    logger.info("[init_db.init_milvus] Schema已创建，共13个字段")

    # 5. 创建索引
    nlist = Config.config.getint('retrieval', 'ivf_nlist', fallback=128)
    drop_ratio = Config.config.getfloat('retrieval', 'drop_ratio_build', fallback=0.2)

    index_params = client.prepare_index_params()

    # 稠密向量索引: IVF_FLAT + IP(内积)
    index_params.add_index(
        field_name="dense_vector",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": nlist}
    )
    logger.info(f"[init_db.init_milvus] 稠密向量索引: IVF_FLAT, nlist={nlist}, IP")

    # 稀疏向量索引: SPARSE_INVERTED_INDEX
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_build": drop_ratio}
    )
    logger.info(f"[init_db.init_milvus] 稀疏向量索引: SPARSE_INVERTED_INDEX, drop_ratio={drop_ratio}")

    # 6. 创建集合（带索引）
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info(f"[init_db.init_milvus] 集合 `{collection_name}` 已创建")

    client.close()
    logger.info("[init_db.init_milvus] Milvus初始化完成")


# ==================== 主入口 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("  EduRAG管理学(河南)在线问答系统 - 数据库初始化")
    print("=" * 60)

    try:
        init_mysql()
        print("\n[OK] MySQL 初始化成功\n")
    except Exception as e:
        print(f"\n[FAIL] MySQL 初始化失败: {e}\n")
        logger.error(f"[init_db.main] MySQL初始化失败: {e}")

    try:
        init_milvus()
        print("\n[OK] Milvus 初始化成功\n")
    except Exception as e:
        print(f"\n[FAIL] Milvus 初始化失败: {e}\n")
        logger.error(f"[init_db.main] Milvus初始化失败: {e}")

    print("=" * 60)
    print("  数据库初始化完成!")
    print("=" * 60)
