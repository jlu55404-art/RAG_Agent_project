"""
脚本作用:
    主要用来处理 PDF文件，帮我们把 PDF 里的文字内容提取出来，尤其是那些包含图片或扫描内容的 PDF(比如老师的手写课件扫描件、带公式图片的教材 PDF)
这个 OCRPDFLoader 的核心功能：
    先提取 PDF 里本来就有的文字（比如直接用电脑编辑的 PDF 文本）
    遇到 PDF 里的图片时，会先判断图片大小（过滤掉太小的图片，提高处理速度）
    对符合条件的图片，用 OCR 技术识别里面的文字
    还能处理旋转的页面，确保图片文字识别准确
    逐页返回 Document，每个 Document 携带 page_number 和 chapter 元数据
    自动识别章节标题（如"第X章 xxx"、"第X节 xxx"），为后续文本块标记章节名
特别适合处理教育场景中常见的复杂 PDF（比如混合了文字、公式图片、手写批注的课件），让这些 "藏" 在图片里的内容都能被系统识别和利用。
"""


import re  # 用于正则匹配章节标题
import cv2  # OpenCV 库，用于处理图像（比如旋转图片）
import fitz  # pyMuPDF里面的fitz包，不要与pip install fitz混淆  # fitz 是 PyMuPDF 库的核心模块，用来读取和解析 PDF 文件
import numpy as np  # numpy 用于处理数组数据，比如把图片转成矩阵
from PIL import Image  # PIL 是图像处理库，这里用来打开和转换图片格式
from tqdm import tqdm  # tqdm 是进度条库，用来显示处理进度
from typing import Iterator  # Iterator 是 Python 类型提示，表示一个迭代器（可以逐个产出数据）
from langchain_core.documents import Document  # LangChain 框架的 Document 类，用来封装提取出的文本和元数据
from langchain_core.document_loaders import BaseLoader  # LangChain 框架的 BaseLoader 基类，所有文档加载器都要继承它

from _edu_ocr import get_ocr  # 导入本地 OCR 模块，get_ocr() 返回一个能识别图片中文字的函数

# PDF OCR 控制：只对宽高超过页面一定比例（图片宽/页面宽，图片高/页面高）的图片进行 OCR。
# 这样可以避免 PDF 中一些小图片的干扰，提高非扫描版 PDF 处理速度
PDF_OCR_THRESHOLD = (0.6, 0.6)

# 章节标题匹配模式：支持"第X章"、"第X节"、"第X部分"、"第X篇"等中文编号，以及阿拉伯数字编号
# 匹配示例："第一章 绪论"、"第3节 实验方法"、"第二部分 实践"
_CHAPTER_PATTERN = re.compile(
    r'^\s*'                                          # 行首允许空白
    r'(第[一二三四五六七八九十百千零\d]+[章节省部分篇])'  # "第X章" 等
    r'[\s：:]*'                                      # 标题与正文间的分隔符
    r'(.+)?'                                         # 章节名称（可选）
)


def _detect_chapter(text: str):
    """从文本中检测章节标题。
    逐行扫描，返回第一个匹配到的章节标题字符串（如 "第一章 绪论"），未匹配则返回 None。
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _CHAPTER_PATTERN.match(line)
        if m:
            prefix = m.group(1)
            suffix = (m.group(2) or "").strip()
            return f"{prefix} {suffix}".strip() if suffix else prefix
    return None


class OCRPDFLoader(BaseLoader):
    """PDF 文档加载器，逐页提取文本并携带页码和章节元数据。"""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """逐页加载 PDF，每页产出一个 Document，metadata 中包含 page_number 和 chapter。"""
        yield from self._pdf2documents()

    def _pdf2documents(self) -> Iterator[Document]:
        """核心方法：逐页提取 PDF 文本，追踪章节信息，每页返回一个 Document。"""
        ocr = get_ocr()
        doc = fitz.open(self.file_path)
        current_chapter = ""  # 当前章节名，随页面推进动态更新

        b_unit = tqdm(total=doc.page_count, desc="OCRPDFLoader context page index: 0")
        for i, page in enumerate(doc):
            page_number = i + 1  # 页码从 1 开始
            b_unit.set_description("OCRPDFLoader context page index: {}".format(i))
            b_unit.refresh()

            # 提取文本：默认使用 "text" 模式提取文本
            text = page.get_text("text")

            # 获取图片并做 OCR 识别，保留原有的 OCR 和旋转处理逻辑
            img_list = page.get_image_info(xrefs=True)
            for img in img_list:
                if xref := img.get("xref"):
                    bbox = img["bbox"]
                    # 检查图片尺寸是否超过设定的阈值，太小的图片跳过
                    if ((bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0]
                            or (bbox[3] - bbox[1]) / (page.rect.height) < PDF_OCR_THRESHOLD[1]):
                        continue
                    pix = fitz.Pixmap(doc, xref)
                    # 如果 Page 有旋转角度，则旋转图片以保证 OCR 识别准确
                    if int(page.rotation) != 0:
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                        tmp_img = Image.fromarray(img_array)
                        ori_img = cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR)
                        rot_img = self.rotate_img(img=ori_img, angle=360 - page.rotation)
                        img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                    else:
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)

                    result, _ = ocr(img_array)
                    if result:
                        ocr_result = [line[1] for line in result]
                        text += "\n" + "\n".join(ocr_result)

            # 尝试从当前页文本中检测章节标题
            detected = _detect_chapter(text)
            if detected:
                current_chapter = detected

            # 跳过完全空白的页面
            if not text.strip():
                b_unit.update(1)
                continue

            yield Document(
                page_content=text,
                metadata={
                    "source": self.file_path,
                    "page_number": page_number,
                    "chapter": current_chapter,
                },
            )

            b_unit.update(1)

    def rotate_img(self, img, angle):
        '''
        img   --image
        angle --rotation angle
        return--rotated img
        '''
        h, w = img.shape[:2]
        rotate_center = (w / 2, h / 2)
        # 获取旋转矩阵
        # 参数1为旋转中心点;
        # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
        # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        # 计算图像新边界
        new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
        new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
        return rotated_img

if __name__ == '__main__':
    pdf_loader = OCRPDFLoader(file_path="samples/ocr_03.pdf")
    docs = pdf_loader.load()

    for d in docs:
        print(f"page={d.metadata.get('page_number')}, "
              f"chapter={d.metadata.get('chapter')}, "
              f"content_len={len(d.page_content)}")
