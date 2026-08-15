"""
EduRAG管理学(河南)在线问答系统 - 日志系统
格式: [级别] 时间 [模块.类.函数:行号] 消息
原则: 不重复、不丢失、可追溯
"""
import os
import logging
import inspect
from datetime import datetime
from typing import Optional

from base.config import Config


# 日志目录
LOG_ROOT = Config.LOG_DIR
os.makedirs(LOG_ROOT, exist_ok=True)


class CallerInfoFilter(logging.Filter):
    """自动提取调用者信息的过滤器
    格式: [模块名.类名.函数名:行号]
    """

    def filter(self, record):
        # 获取调用栈，跳过 logging 内部帧和 logger 自身的方法帧
        frame = inspect.currentframe()
        try:
            # 向上追溯调用栈，找到真正的调用者
            caller_frame = frame
            for _ in range(10):  # 最多向上10层
                caller_frame = caller_frame.f_back
                if caller_frame is None:
                    break
                module_name = caller_frame.f_globals.get('__name__', '')
                # 跳过 logging 和 base.logger 模块
                if module_name and not module_name.startswith('logging') and module_name != 'base.logger':
                    break

            if caller_frame:
                module_name = caller_frame.f_globals.get('__name__', 'unknown')
                # 只取模块名的最后一段（如 rag_qa.core.vector_store → vector_store）
                module_short = module_name.split('.')[-1] if '.' in module_name else module_name

                # 尝试获取类名
                class_name = ''
                if 'self' in caller_frame.f_locals:
                    class_name = caller_frame.f_locals['self'].__class__.__name__
                elif 'cls' in caller_frame.f_locals:
                    class_name = caller_frame.f_locals['cls'].__name__

                func_name = caller_frame.f_code.co_name
                line_no = caller_frame.f_lineno

                if class_name:
                    record.caller_info = f"{module_short}.{class_name}.{func_name}:{line_no}"
                else:
                    record.caller_info = f"{module_short}.{func_name}:{line_no}"
            else:
                record.caller_info = "unknown"
        finally:
            del frame

        return True


class EduRAGFormatter(logging.Formatter):
    """自定义日志格式器
    输出格式: [级别] 时间 [模块.类.函数:行号] 消息
    """

    def __init__(self):
        super().__init__(
            fmt="[%(levelname)s] %(asctime)s [%(caller_info)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def get_logger(
        name: str = 'ManagementRAG',
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file: Optional[str] = None,
) -> logging.Logger:
    """创建并配置日志对象"""
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)

    # 防止重复添加handler
    if _logger.handlers:
        return _logger

    # 添加调用者信息过滤器
    caller_filter = CallerInfoFilter()
    formatter = EduRAGFormatter()

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(caller_filter)
    _logger.addHandler(console_handler)

    # 文件handler
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f'{datetime.now().strftime("%Y%m%d")}.log')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(caller_filter)
    _logger.addHandler(file_handler)

    return _logger


# 全局唯一日志对象
logger = get_logger(
    console_level=Config.LOG_CONSOLE_LEVEL,
    file_level=Config.LOG_FILE_LEVEL,
)


if __name__ == '__main__':
    logger.debug('调试信息测试')
    logger.info('普通信息测试')
    logger.warning('警告信息测试')
    logger.error('错误信息测试')
    logger.critical('严重错误测试')
