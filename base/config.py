"""
EduRAG管理学(河南)在线问答系统 - 全局配置
所有配置统一从 config.ini 读取，禁止硬编码
"""
import configparser
import json
import os
from base.path_tools import get_abs_path


class Config:
    """全局配置类（单例模式）"""
    CONFIG_REL_NAME = 'config.ini'

    def __init__(self, config_file=None):
        if config_file is None:
            config_file = get_abs_path(Config.CONFIG_REL_NAME)

        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8')

        # ── 日志配置 ──
        self._LOG_REL_DIR = self.config.get('log', 'log_dir')
        self.LOG_DIR = get_abs_path(self._LOG_REL_DIR)
        self.LOG_CONSOLE_LEVEL = self.config.getint('log', 'log_console_level', fallback=20)
        self.LOG_FILE_LEVEL = self.config.getint('log', 'log_file_level', fallback=20)

        # ── MySQL配置 ──
        self.MYSQL_HOST = self.config.get('mysql', 'host')
        self.MYSQL_PORT = self.config.getint('mysql', 'port')
        self.MYSQL_USER = self.config.get('mysql', 'user')
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password')
        self.MYSQL_DB = self.config.get('mysql', 'database')
        self.MYSQL_TABLE = self.config.get('mysql', 'table')

        # ── Redis配置 ──
        self.REDIS_HOST = self.config.get('redis', 'host')
        self.REDIS_PORT = self.config.getint('redis', 'port')
        self.REDIS_PASSWORD = self.config.get('redis', 'password')
        self.REDIS_DB = self.config.getint('redis', 'db')
        self.REDIS_TTL = self.config.getint('redis', 'ttl')

        # ── Milvus配置 ──
        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='127.0.0.1')
        self.MILVUS_PORT = self.config.getint('milvus', 'port', fallback=19530)
        self.MILVUS_DATABASE = self.config.get('milvus', 'database_name', fallback='default')
        self.MILVUS_COLLECTION = self.config.get('milvus', 'collection_name', fallback='test')

        # ── 数据配置 ──
        self.DATA_DIR = get_abs_path(self.config.get('data', 'data_dir', fallback='data'))

        # ── 模型配置 ──
        self.LLM_MODEL = self.config.get('model', 'dashscope_model', fallback='qwen3.7-max')

        try:
            self.DASHSCOPE_API_KEY = self.config.get('model', 'dashscope_api_key')
        except Exception:
            self.DASHSCOPE_API_KEY = os.getenv(self.config.get('model', 'env_api_key', fallback=''))

        self.DASHSCOPE_BASE_URL = self.config.get('model', 'dashscope_base_url', fallback='')
        self.EMBEDDING_MODEL_PATH = get_abs_path(self.config.get('model', 'embedding_path', fallback=''))
        self.RERANK_MODEL_PATH = get_abs_path(self.config.get('model', 'rerank_path', fallback=''))

        # ── 检索参数 ──
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=6)
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=3)
        self.BM25_THRESHOLD = self.config.getfloat('retrieval', 'bm25_threshold', fallback=0.85)
        self.DENSE_WEIGHT = self.config.getfloat('retrieval', 'dense_weight', fallback=1.0)
        self.SPARSE_WEIGHT = self.config.getfloat('retrieval', 'sparse_weight', fallback=0.7)
        self.DROP_RATIO_BUILD = self.config.getfloat('retrieval', 'drop_ratio_build', fallback=0.2)
        self.DROP_RATIO_SEARCH = self.config.getfloat('retrieval', 'drop_ratio_search', fallback=0.2)
        self.IVF_NLIST = self.config.getint('retrieval', 'ivf_nlist', fallback=128)
        self.NPROBE = self.config.getint('retrieval', 'nprobe', fallback=10)

        # ── 应用配置 ──
        self.VALID_SOURCES = json.loads(
            self.config.get('app', 'valid_sources', fallback='["management"]'))
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='***********')
        self.DEFAULT_QUOTA = self.config.getint('app', 'default_quota', fallback=50)
        self.JWT_SECRET_KEY = self.config.get('app', 'jwt_secret_key', fallback='***********')
        self.JWT_EXPIRE_HOURS = self.config.getint('app', 'jwt_expire_hours', fallback=24)

        # ── 设备配置 ──
        self.DEVICE = self.config.get('device', 'device', fallback='cpu')


# 全局单例
Config = Config()


if __name__ == '__main__':
    print(f"MySQL: {Config.MYSQL_HOST}:{Config.MYSQL_PORT}/{Config.MYSQL_DB}")
    print(f"Milvus: {Config.MILVUS_HOST}:{Config.MILVUS_PORT}/{Config.MILVUS_DATABASE}")
    print(f"LLM: {Config.LLM_MODEL}")
    print(f"Sources: {Config.VALID_SOURCES}")
    print(f"Retrieval: K={Config.RETRIEVAL_K}, M={Config.CANDIDATE_M}")
