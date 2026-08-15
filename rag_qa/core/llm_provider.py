# todo 这个文件是LLM大模型的"供应商"模块，负责创建和管理大模型对象
# todo 用了"单例模式"——整个项目只创建一个大模型对象，所有人共享，避免重复创建浪费资源
# todo ChatTongyi是阿里通义千问大模型的API封装，通过这个对象我们可以跟大模型对话

# todo 从langchain_community导入通义千问的聊天模型类
from langchain_community.chat_models.tongyi import ChatTongyi
# todo base是本项目的公共基础模块，提供日志(logger)和配置(Config)
from base import logger, Config

# 全局单例：保证整个项目只创建一个LLM对象
# todo _llm_instance是一个全局变量，存着那个"唯一的大模型对象"，初始为None表示还没创建
_llm_instance = None


def get_llm():
    """获取云上LLM大模型对象(单例模式, 全局只创建一次)
    :return: ChatTongyi 大模型对象
    """
    # todo 这个函数就是"单例模式"的经典写法：
    # todo   第1次调用：发现_llm_instance是None → 创建新对象 → 保存到_llm_instance → 返回
    # todo   第2次及以后：发现_llm_instance已经有了 → 直接返回，不重复创建
    global _llm_instance # 声明要修改使用全局变量 _llm_instance
    # 如果全局变量 _llm_instance 为 None，则创建一个
    if _llm_instance is None:
        # 创建一个ChatTongyi对象
        # todo ChatTongyi就是通义千问大模型的"遥控器"，通过它发消息给大模型
        _llm_instance = ChatTongyi(
            # todo api_key相当于"登录密码"，告诉阿里云你是谁
            api_key=Config.DASHSCOPE_API_KEY,
            # todo model指定用哪个大模型版本，比如qwen-turbo、qwen-plus等
            model=Config.LLM_MODEL,
            # todo temperature控制回答的"创造性"，0.1很低→回答更保守/稳定，1.0很高→回答更天马行空
            temperature=0.1,
        )
        logger.info(f'LLM实例已创建: model={Config.LLM_MODEL}')
    return _llm_instance
