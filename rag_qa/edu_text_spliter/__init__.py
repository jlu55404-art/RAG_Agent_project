
# todo 这个文件是"文本分割器"模块的入口，负责把所有文本分割工具统一导出
# todo 为什么要分割文本？因为大模型一次能处理的文字有限（有"上下文窗口"限制），而且大段文本一股脑丢给模型效果不好
# todo 所以我们要把长文档切成一小块一小块的"片段"，每个片段都是独立的语义单元，方便检索和问答

import sys, os
# todo 获取当前文件所在的文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# todo 把当前目录加到Python搜索路径最前面，确保能导入同目录下的分割器模块
sys.path.insert(0, current_dir)

# todo AliTextSplitter：阿里开源的文本分割器，根据语义边界来切分，而不是生硬地按字数切
# todo SimpleAliTextSplitter：阿里分割器的简化版，更快但可能没那么精细
from edu_model_text_spliter import AliTextSplitter, SimpleAliTextSplitter
# todo RecursiveCharacterTextSplitter：递归字符分割器，按优先级用不同分隔符（换行→句号→空格→字符）逐层切分
# todo ChineseRecursiveTextSplitter：中文专用递归分割器，针对中文做了优化，优先按中文标点切分
from edu_chinese_recursive_text_splitter import RecursiveCharacterTextSplitter, ChineseRecursiveTextSplitter

# todo MarkdownTextSplitter：专门处理Markdown格式的分割器，会保留标题层级，不会把标题和内容拆散
from langchain_text_splitters import MarkdownTextSplitter

# todo 统一导出的分割器列表
__all__ = [
    "AliTextSplitter",
    "SimpleAliTextSplitter",
    "RecursiveCharacterTextSplitter",
    "ChineseRecursiveTextSplitter",
    "MarkdownTextSplitter",
]
