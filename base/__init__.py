
# todo __init__.py 的作用：当你 import base 这个包时，Python会自动执行这个文件
# todo 这里把常用的工具提前导入，其他文件直接写 from base import Config 就能用，不用写一长串路径

from .config import Config # 全局唯一配置对象  todo .config 就是 base/config.py
from .logger import logger # 全局唯一日志对象   todo .logger 就是 base/logger.py
from .path_tools import get_abs_path # 获取绝对路径方法  todo 把相对路径变成绝对路径
from .path_tools import get_project_root # 获取项目根路径方法  todo 无论在哪里运行都能找到项目根目录

''' 统一导出模式
在项目中其他文件里，直接可以这样写⬇️
from base import Config
好处是：
1. 包内文件代码随便改，只要改自己的__init__.py就可以了
2. 项目中其他代码无须做任何改变、不受任何影响
'''

__all__ = [
    'Config',
    'logger',
    'get_abs_path',
    'get_project_root'
]
