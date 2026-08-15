from typing import Iterator  # todo Iterator 是 Python 类型提示，表示迭代器（可以逐个产出数据）
from langchain_core.documents import Document  # todo LangChain 的 Document 类，封装提取出的文本和来源信息
from langchain_core.document_loaders import BaseLoader  # todo LangChain 的 BaseLoader 基类，所有文档加载器都继承它
from pptx import Presentation  # todo python-pptx 库，用来读取 PowerPoint（.pptx）文件
from PIL import Image  # todo PIL 图像处理库，把二进制数据转成图片对象
import numpy as np  # todo numpy 把图片转成数组给 OCR 识别
from io import BytesIO  # todo BytesIO 在内存中模拟文件读写操作
from tqdm import tqdm  # todo tqdm 是进度条库，显示处理进度

from _edu_ocr import get_ocr  # todo 导入本地 OCR 模块


class OCRPPTLoader(BaseLoader):
    """An example document loader that reads a file line by line."""
    # todo 这个类的功能：读取 PowerPoint 文件，提取每张幻灯片中的文字、表格和图片中的文字
    # todo 继承自 LangChain 的 BaseLoader，可以被 LangChain 框架统一调用

    def __init__(self, filepath: str) -> None:
        """Initialize the loader with a file path.

        Args:
            filepath: The path to the ppt to load.
        """
        # todo 初始化函数，保存 PPT 文件路径
        self.filepath = filepath

    def lazy_load(self) -> Iterator[Document]:
        # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        # todo "懒加载"方法：用 yield 方式产出文档内容，省内存

        line = self.ppt2text(self.filepath)  # todo 调用 ppt2text 把 PPT 转成纯文本
        yield Document(page_content=line, metadata={"source": self.filepath})  # todo 包装成 Document 对象返回

    def ppt2text(self, filepath):
        # todo 核心方法：读取 PPT 中所有幻灯片的文字、表格和图片，提取全部文本
        # 打开指定路径的 PowerPoint 文件
        prs = Presentation(filepath)  # todo 用 python-pptx 库打开 PPT 文件
        print(f'prs-->{prs}')
        # 获取 OCR 功能的实例
        ocr = get_ocr()  # todo 获取 OCR 识别函数
        # 初始化一个空字符串，用于存储提取的文本内容
        resp = ""  # todo 用来收集所有提取到的文字

        def extract_text(shape):
            # nonlocal指明resp非全局非局部，而是外部嵌套函数中的变量，
            # 允许内部函数访问和修改外部函数中定义的变量resp
            nonlocal resp  # todo nonlocal 关键字：让内部函数能修改外部函数的 resp 变量

            # 检查形状是否有文本框
            if shape.has_text_frame:
                # 将文本框中的文本添加到resp中，并去掉前后空格
                resp += shape.text.strip() + "\n"  # todo 如果这个形状是文本框，直接提取里面的文字

            # 检查形状是否为表格
            if shape.has_table:
                # 遍历表格的每一行
                for row in shape.table.rows:
                    # 遍历每一行中的每个单元格
                    for cell in row.cells:
                        # 遍历单元格中的每个段落
                        for paragraph in cell.text_frame.paragraphs:
                            # 将单元格中的文本添加到resp中，并去掉前后空格
                            resp += paragraph.text.strip() + "\n"  # todo 把表格每个单元格里的文字也提取出来

            # 检查形状是否为图片（shape_type == 13）
            if shape.shape_type == 13:  # 13 表示图片  # todo shape_type==13 代表这是一个图片形状
                # 使用 BytesIO 打开图片数据并转换为图像对象
                image = Image.open(BytesIO(shape.image.blob))  # todo 把图片的二进制数据转成图片对象
                # 使用 OCR 处理图像并获取结果
                result, _ = ocr(np.array(image))  # todo 把图片转成 numpy 数组，交给 OCR 识别
                if result:  # 如果 OCR 有结果
                    # 提取 OCR 结果中的文本行
                    ocr_result = [line[1] for line in result]  # todo 从识别结果中提取文字内容
                    # 将 OCR 提取的文本添加到resp中，以换行分隔
                    resp += "\n".join(ocr_result)  # todo 把识别出的文字拼接到结果中

            # 检查形状是否为组合形状（shape_type == 6）
            elif shape.shape_type == 6:  # 6 表示组合  # todo shape_type==6 代表这是组合形状（多个形状打包在一起）
                # 遍历组合形状中的每个子形状，递归调用extract_text函数
                for child_shape in shape.shapes:
                    extract_text(child_shape)  # todo 递归处理组合形状里的每个子形状

        # 创建一个进度条，用于显示幻灯片处理进度，初始总数为幻灯片数量
        b_unit = tqdm(total=len(prs.slides), desc="OCRPPTLoader slide index: 1")  # todo 进度条总量 = 幻灯片数量

        # 遍历所有幻灯片
        for slide_number, slide in enumerate(prs.slides, start=1):
            # 更新进度条描述，显示当前处理的幻灯片索引
            b_unit.set_description("OCRPPTLoader slide index: {}".format(slide_number))
            b_unit.refresh()  # 刷新进度条显示

            # 按照从上到下、从左到右的顺序对形状进行排序遍历
            sorted_shapes = sorted(slide.shapes, key=lambda x: (x.top, x.left))  # todo 把幻灯片上的形状按位置排序（先从上到下，再从左到右），这样提取顺序更自然

            for shape in sorted_shapes:
                extract_text(shape)  # 调用extract_text函数提取当前形状的文本内容  # todo 对每个形状调用 extract_text 提取文字

            b_unit.update(1)  # 更新进度条，表示处理了一张幻灯片  # todo 进度条走一步

        return resp  # 返回提取到的所有文本内容  # todo 返回全部提取的文字


if __name__ == '__main__':
    img_loader = OCRPPTLoader(filepath='samples/ocr_01.pptx')  # todo 测试用：加载示例 PPT 文件
    doc = img_loader.load()  # todo 调用 load() 提取 PPT 中的所有内容
    print(doc)