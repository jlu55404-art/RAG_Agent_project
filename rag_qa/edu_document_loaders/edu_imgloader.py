from typing import Iterator  # todo Iterator 是 Python 类型提示，表示迭代器（可以逐个产出数据）
from langchain_core.documents import Document  # todo LangChain 的 Document 类，封装提取出的文本和来源
from langchain_core.document_loaders import BaseLoader  # todo LangChain 的 BaseLoader 基类，所有文档加载器都继承它

from _edu_ocr import get_ocr  # todo 导入本地 OCR 模块，获取文字识别函数

class OCRIMGLoader(BaseLoader):
    """An example document loader that reads a file line by line."""
    # todo 这个类的功能：读取单张图片文件，用 OCR 识别图片中的文字并返回
    # todo 继承自 LangChain 的 BaseLoader，可以被 LangChain 框架统一调用

    def __init__(self, img_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            img_path: The path to the img to load.
        """
        # todo 初始化函数，保存图片文件路径
        self.img_path = img_path

    def lazy_load(self) -> Iterator[Document]:
        # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        # todo "懒加载"方法：用 yield 一条一条产出文档数据

        line = self.img2text()  # todo 调用 img2text 方法，识别图片中的文字
        yield Document(page_content=line, metadata={"source": self.img_path})  # todo 把识别出的文字和图片路径打包成 Document 返回

    def img2text(self):
        # todo 核心方法：读取图片文件，用 OCR 识别里面的文字
        resp = ""  # todo 用来收集 OCR 识别出的文字
        ocr = get_ocr()  # todo 获取 OCR 识别函数
        result, _ = ocr(self.img_path)  # todo 对图片执行 OCR 识别，result 是识别结果列表
        if result:
            ocr_result = [line[1] for line in result]  # todo 从识别结果中提取文字（每个结果的第2个元素是文字内容）
            resp += "\n".join(ocr_result)  # todo 用换行符拼接所有识别出的文字行
        return resp  # todo 返回识别到的全部文字


if __name__ == '__main__':
    img_loader = OCRIMGLoader(img_path='samples/ocr_04.png')  # todo 测试用：加载示例图片文件
    doc = img_loader.load()  # todo 调用 load() 识别图片中的文字
    print(doc)