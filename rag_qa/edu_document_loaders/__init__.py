# todo 这个文件是"文档加载器"模块的入口，负责把所有文档加载相关的工具统一导出
# todo 文档加载器的作用：把各种格式的文档（txt、md、pdf、图片、ppt等）读入程序，变成程序能处理的文本
# todo 就像"万能开瓶器"——不同格式的文件用不同的加载器，但最终都得到文字内容

# 添加当前目录到系统路径
# todo sys.path是Python找模块时会搜索的路径列表，把自己目录加进去，确保能import到同目录下的其他文件
import sys, os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)

# todo TextLoader：加载纯文本.txt文件
from langchain_community.document_loaders import TextLoader
# todo UnstructuredMarkdownLoader：加载Markdown.md文件，会保留标题层级等结构
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

# todo OCRDOCLoader：加载.doc文档，用OCR技术识别里面的文字（比如扫描版文档也能读）
from edu_docloader import OCRDOCLoader
# todo OCRPPTLoader：加载.ppt幻灯片，同样用OCR识别文字
from edu_pptloader import OCRPPTLoader
# todo OCRIMGLoader：加载图片文件（如jpg/png），用OCR把图片里的文字"读"出来
from edu_imgloader import OCRIMGLoader
# todo OCRPDFLoader：加载PDF文件，用OCR识别文字（比普通PDF读取更强大，能处理扫描版PDF）
from edu_pdfloader import OCRPDFLoader

# todo __all__定义了当你写 from xxx import * 时，哪些名字会被导出，相当于"对外开放的菜单"
__all__ = [
    "TextLoader",
    "UnstructuredMarkdownLoader",
    "OCRDOCLoader",
    "OCRPPTLoader",
    "OCRIMGLoader",
    "OCRPDFLoader",
]
