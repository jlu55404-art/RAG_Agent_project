
# todo 这个文件是OCR（光学字符识别）引擎的获取模块
# todo 简单说OCR就是"让电脑看懂图片里的字"——拍张照片，OCR能告诉你照片上写了什么文字
# todo 这个项目用它来加载扫描版PDF、图片、PPT等无法直接复制文字的文档

'''
啥是 OCR(Optical Character Recognition)?   可以直接问 大模型, 让他给我们解释下.
    简单说就是"光学字符识别"技术，能让电脑"看懂"图片里的文字，把图片上的手写体、印刷体文字转换成可编辑的文本(比如照片里的笔记、扫描版 PDF 里的文字)

paddleocr：解析图片中的文字，也可以进行表格识别

rapidocr_paddle 和 rapidocr_onnxruntime 两种导入方式
    主要区别在于它们所使用的推理引擎和硬件支持
    选择哪种方式最合适取决于你的硬件环境和性能需求。

    当你有 GPU 且追求速度时：使用 rapidocr_paddle。
    PaddlePaddle 原生支持在 GPU 上推理 PaddleOCR 模型，速度更快。
    当只有 CPU 且需要高效推理时：使用 rapidocr_onnxruntime。
    它在 CPU 上进行了优化，资源占用较低.
'''

def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    # todo 获取OCR引擎对象，优先用GPU加速版，不行就降级到CPU版
    # todo use_cuda参数控制是否启用GPU加速：True=用显卡加速（快），False=纯CPU跑（慢但兼容性好）
    """
    :return : paddleocr图像识别对象
    """
    try:
        # todo 先尝试导入GPU加速版的rapidocr_paddle，需要电脑有NVIDIA显卡和CUDA环境
        from rapidocr_paddle import RapidOCR
        '''
        det_use_cuda=True：启用检测模型的GPU加速。cls_use_cuda=True：启用分类模型的GPU加速。rec_use_cuda=True：启用识别模型的GPU加速。
        '''
        # todo OCR分三个步骤：1.det检测文字位置 2.cls判断文字方向 3.rec识别文字内容
        # todo 三个use_cuda参数分别控制这三个步骤是否用GPU加速
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        # todo 如果GPU版导入失败（比如没有装CUDA或没有显卡），自动降级到CPU版的onnxruntime
        from rapidocr_onnxruntime import RapidOCR
        # todo CPU版不需要use_cuda参数，纯用CPU计算
        ocr = RapidOCR()
    # todo 返回OCR对象，调用方用它来识别图片里的文字
    return ocr
