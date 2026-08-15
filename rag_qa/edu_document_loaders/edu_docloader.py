from typing import Iterator  # todo Iterator 是 Python 类型提示，表示一个迭代器（可以逐个产出数据）
from tqdm import tqdm  # todo tqdm 是进度条库，用来显示处理进度
from docx.table import _Cell, Table  # 用于处理表格
from docx.oxml.table import CT_Tbl  # 用于处理表格XML结构
from docx.oxml.text.paragraph import CT_P  # 用于处理段落XML结构
from docx.text.paragraph import Paragraph  # 用于处理段落内容
from docx import Document as Docu1  # todo Docu1 是 python-docx 库的 Document 类，用来读取 Word 文档
from docx.document import Document as Docu2  # todo Docu2 是 python-docx 内部的实际文档对象类型，用于类型判断
from docx import ImagePart  # 用于处理Word文档和图片  # todo ImagePart 表示 Word 文档中嵌入的图片
from PIL import Image  # 用于处理图片  # todo PIL 是图像处理库，这里用来把字节数据转成图片对象
from io import BytesIO  # 用于将字节流转换为图片  # todo BytesIO 在内存中模拟文件读写，把二进制数据当文件来操作
import numpy as np  # 用于处理数组  # todo numpy 用于把图片转成数组给 OCR 识别
from langchain_core.documents import Document  # todo LangChain 的 Document 类，封装提取出的文本和来源信息
from langchain_core.document_loaders import BaseLoader  # todo LangChain 的 BaseLoader 基类，所有文档加载器都继承这个

from _edu_ocr import get_ocr  # todo 导入本地 OCR 模块，拿到文字识别函数


class OCRDOCLoader(BaseLoader):
    """An example document loader that reads a file line by line."""
    # todo 这个类的功能：读取 Word（.docx）文件，提取里面的文字、表格内容和图片中的文字
    # todo 是 LangChain 的 BaseLoader 子类，可以被 LangChain 框架统一调用

    def __init__(self, filepath: str) -> None:
        """Initialize the loader with a file path.

        Args:
            filepath_path: The path to the filepath to load.
        """
        # todo 初始化函数，接收 Word 文件路径，存起来备用
        self.filepath = filepath

    def lazy_load(self) -> Iterator[Document]:
        # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        # todo "懒加载"方法：用 yield 的方式把文档内容一条一条产出，省内存

        line = self.doc2text(self.filepath)  # todo 调用 doc2text 方法，把 Word 文件转成纯文本
        yield Document(page_content=line, metadata={"source": self.filepath})  # todo 包装成 Document 对象返回

    def doc2text(self, filepath):
        # todo 核心方法：把 Word 文件里的段落文字、表格文字、嵌入图片中的文字全部提取出来

        # 创建OCR识别对象
        ocr = get_ocr()  # todo 获取 OCR 识别函数
        # print(f'ocr--》{ocr}')  # 输出OCR对象信息

        # 读取Word文档
        doc = Docu1(filepath)  # todo 用 python-docx 打开 Word 文件
        # print(f'doc-->{doc}')  # 输出读取到的文档信息
        # 定义一个空字符串用于存储最终的文本内容
        resp = ""  # todo 用来收集所有提取到的文字
        # 定义一个迭代器，用于遍历文档中的块（段落、表格等）
        def iter_block_items(parent):
            # todo 这个内部函数的作用：逐个遍历 Word 文档中的"块"（段落或者表格）
            # todo parent 可以是整个文档对象，也可以是表格中的某个单元格
            # 判断parent对象类型，如果是Document类型，则获取其元素
            if isinstance(parent, Docu2):
                parent_elm = parent.element.body  # todo 如果是整个文档，拿到文档的 body 元素
            # 如果是表格单元格类型，获取单元格的XML元素
            elif isinstance(parent, _Cell):
                parent_elm = parent._tc  # todo 如果是单元格，拿到单元格的 XML 元素
            else:
                raise ValueError("OCRDOCLoader parse fail")  # 如果都不是，则抛出错误
            # print(f'parent_elm--》{parent_elm}')
            # print('*'*80)
            # 遍历parent_elm中的所有子元素
            for child in parent_elm.iterchildren():  # todo 遍历所有子元素（XML 层级下的每个子节点）
                # print(f'child--》{child}')
                if isinstance(child, CT_P):  # 如果是段落类型
                    yield Paragraph(child, parent)  # 返回段落  # todo CT_P 是段落的 XML 标签，说明这是一段文字
                elif isinstance(child, CT_Tbl):  # 如果是表格类型
                    yield Table(child, parent)  # 返回表格  # todo CT_Tbl 是表格的 XML 标签，说明这是一个表格

        # print(f'doc.paragraphs-->{doc.paragraphs}')
        # print(f'doc.tables-->{doc.tables}')
        # 创建进度条，表示文档处理的进度
        b_unit = tqdm(total=len(doc.paragraphs) + len(doc.tables),
                      desc="OCRDOCLoader block index: 0")  # todo 进度条总量 = 段落数 + 表格数

        # 遍历文档中的所有块（段落和表格）
        for i, block in enumerate(iter_block_items(doc)):
            # 更新进度条描述
            b_unit.set_description("OCRDOCLoader  block index: {}".format(i))
            b_unit.refresh()  # 刷新进度条

            # 如果块是段落类型
            if isinstance(block, Paragraph):
                resp += block.text.strip() + "\n"  # 将段落文本加入到返回字符串中  # todo 先提取段落中的文字
                # 获取段落中的所有图片
                images = block._element.xpath('.//pic:pic')  # todo 用 XPath 找到段落中嵌入的所有图片
                for image in images:
                    # 遍历图片，获取图片ID
                    for img_id in image.xpath('.//a:blip/@r:embed'):  # todo 获取图片在文档中的引用 ID
                        part = doc.part.related_parts[img_id]  # 根据图片ID获取图片对象  # todo 通过 ID 找到图片的实际数据
                        if isinstance(part, ImagePart):  # 如果该部分是图片  # todo 确认这确实是一张图片
                            # BytesIO 是 Python 内置的 io 模块中的一个类，用于在内存中读写二进制数据
                            # part._blob 通常表示从某个文档（如 DOCX 文件）中提取的二进制内容。
                            image = Image.open(BytesIO(part._blob))  # 打开图片  # todo 把二进制图片数据读成图片对象
                            result, _ = ocr(np.array(image))  # 使用OCR识别图片中的文字  # todo 转成 numpy 数组后丢给 OCR 识别
                            if result:  # 如果识别结果不为空
                                ocr_result = [line[1] for line in result]  # 提取识别出的文字
                                resp += "\n".join(ocr_result)  # 将识别结果加入返回文本中
            # 如果块是表格类型
            elif isinstance(block, Table):
                # 遍历表格中的所有行和单元格
                for row in block.rows:
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            resp += paragraph.text.strip() + "\n"  # 将单元格内的段落文本加入返回文本中  # todo 把每个单元格里的文字也提取出来

            # 更新进度条
            b_unit.update(1)  # todo 进度条走一步
        # 返回提取的文本内容
        return resp  # todo 返回全部提取到的文字



if __name__ == '__main__':
    docx_loader = OCRDOCLoader(filepath='samples/ocr_02.docx')  # todo 测试用：加载示例 Word 文件
    doc = docx_loader.load()  # todo 调用 load() 提取 Word 文档中的所有内容
    print(doc)