# 文本切块
""" TODO 为啥要切父块和子块？
一句话：检索要"准"，生成要"全"，这俩需求天然矛盾，父子块就是用来同时满足它们的。

打个比喻：你去图书馆找"Python装饰器的用法"。
如果整本书是一个大块，你搜"装饰器"，这本书确实命中了，但你得翻遍300页才能找到那2页相关内容——LLM也一样，塞进去1200字的块，真正有用的可能就中间那两句，其余全是噪音，模型容易被无关信息带偏。
如果每句话是一个小块，你搜"装饰器"，精准命中了那一句话，但这句话可能是"它还能带参数"——没有上下文，LLM根本不知道"它"是谁，回答出来前言不搭后语。

父子块就是折中方案：
子块（300字）：相当于书的"索引条目"。每个子块语义聚焦，生成的向量能精确代表这一小段在讲什么。用户问"装饰器怎么带参数"，300字的子块向量跟问题的余弦相似度会非常高，检索命中率极高。
父块（1200字）：相当于"那一整页的内容"。子块命中后，不拿子块去给LLM，而是把它所属的父块（完整上下文）送过去。LLM看到的是有前因后果的完整段落，回答自然连贯、不丢信息。

落到项目代码里的实际流程：

用户提问
  ➡️ 问题转向量
  ➡️ 拿向量去Milvus搜子块（300字，精准匹配）
  ➡️ 命中3个子块，可能来自2个不同的父块
  ➡️ 按parent_id去重，提取2个父块（1200字，完整上下文）
  ➡️ BGE-Reranker对父块重排序
  ➡️ 取Top-M父块塞进Prompt给LLM生成回答

实际意义总结就三点：

1. 检索精度——小块向量语义集中，不会被无关内容稀释，相似度打分更准。
2. 生成质量——大块上下文完整，LLM不会断章取义，回答有前因后果。
3. 去重省token——3个子块可能都属于同一个父块，去重后只送1个父块，避免重复内容浪费LLM的上下文窗口。

如果只用一种块大小，要么检索不准（块太大），要么回答不全（块太小）。父子块把"找得准"和"答得全"拆成两步各自优化，这就是它存在的核心意义。
"""

import os
import re
from datetime import datetime
from base import Config, logger
from langchain_core.documents import Document

from rag_qa.edu_document_loaders import (
    OCRDOCLoader, OCRPDFLoader, OCRIMGLoader,
    OCRPPTLoader, TextLoader, UnstructuredMarkdownLoader
)
from rag_qa.edu_text_spliter import (
    ChineseRecursiveTextSplitter, MarkdownTextSplitter
)

# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": OCRPDFLoader,
    ".docx": OCRDOCLoader,
    ".ppt": OCRPPTLoader,
    ".pptx": OCRPPTLoader,
    ".jpg": OCRIMGLoader,
    ".jpeg": OCRIMGLoader,
    ".png": OCRIMGLoader,
}

# 用于在非 PDF 文档的文本中检测章节标题
_CHAPTER_PATTERN = re.compile(
    r'^\s*'
    r'(第[一二三四五六七八九十百千零\d]+[章节省部分篇])'
    r'[\s：:]*'
    r'(.+)?'
)


def _detect_chapter_from_text(text: str):
    """从文本中检测章节标题，未匹配则返回 None。"""
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


# 定义函数，从指定文件夹加载多种类型文件并添加元数据
def load_documents_from_directory(directory_path):
    """
    加载指定目录下的所有支持的文件类型，并添加元数据。
    :param directory_path: 要处理的文档数据的目录
    :return: [Document]
    """
    documents = []
    supported_extensions = set(document_loaders.keys())

    # 从顶层数据目录名获取学科标签（统一使用顶层目录名，而非子目录名）
    # management_data → management
    source = os.path.basename(directory_path).replace("_data", "")

    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension in supported_extensions:
                try:
                    loader_class = document_loaders[file_extension]
                    if file_extension == ".txt":
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)

                    # 加载文档，返回 [Document,...]
                    loaded_docs = loader.load()

                    # 为每个文档添加元数据
                    for doc in loaded_docs:
                        doc.metadata['source'] = source  # 学科标记（从顶层目录名获取）
                        doc.metadata['file_path'] = file_path  # 文件完整路径
                        doc.metadata['file_name'] = file  # 文件名（不含路径）
                        doc.metadata['version'] = 'v1.0'  # 知识库版本号
                        doc.metadata['timestamp'] = datetime.now().isoformat()

                        # 对于非 PDF 文件，尝试从文本中检测章节信息
                        if not doc.metadata.get('chapter'):
                            detected = _detect_chapter_from_text(doc.page_content)
                            if detected:
                                doc.metadata['chapter'] = detected
                            else:
                                doc.metadata.setdefault('chapter', '')

                        # 对于非 PDF 文件，设置 page_number 默认值
                        doc.metadata.setdefault('page_number', 0)

                    documents.extend(loaded_docs)
                    logger.info(f"[process_documents.load_documents_from_directory] "
                                f"加载[Doc]数量：{len(loaded_docs)} ← from {file_path}")
                except Exception:
                    logger.exception(f"[process_documents.load_documents_from_directory] "
                                     f"加载文件 {file_path} 失败")
            else:
                logger.warning(f"[process_documents.load_documents_from_directory] "
                               f"不支持的文件类型: {file_path}")

    return documents


# 定义函数，处理文档并进行分层切分，返回子块文档对象列表 [Document]
def process_documents(
        directory_path,
        parent_chunk_size=Config.PARENT_CHUNK_SIZE,
        child_chunk_size=Config.CHILD_CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    ) -> [Document]:

    # 加载文档
    documents = load_documents_from_directory(directory_path)

    # 切割器
    parent_splitter = ChineseRecursiveTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=chunk_overlap,
    )
    child_splitter = ChineseRecursiveTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # md专门的切割器
    markdown_parent_splitter = MarkdownTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=chunk_overlap,
    )
    markdown_child_splitter = MarkdownTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 存储子块的列表
    child_chunks = []

    for i, doc in enumerate(documents):
        file_extension = os.path.splitext(doc.metadata['file_path'])[1].lower()
        is_markdown = (file_extension == '.md')

        if is_markdown:
            parent_splitter_to_use = markdown_parent_splitter
        else:
            parent_splitter_to_use = parent_splitter
        if is_markdown:
            child_splitter_to_use = markdown_child_splitter
        else:
            child_splitter_to_use = child_splitter

        # 获取源文档的页码和章节信息（来自 PDF 加载器或非 PDF 的默认值）
        src_page = doc.metadata.get('page_number', 0)
        src_chapter = doc.metadata.get('chapter', '')

        logger.info(f"[process_documents.process_documents] "
                    f"处理文档: {doc.metadata['file_path']}, "
                    f"使用切分器: {'Markdown' if is_markdown else 'ChineseRecursive'}")

        # 切割父块 : [Document]
        parent_docs = parent_splitter_to_use.split_documents([doc])
        for j, parent_doc in enumerate(parent_docs):
            parent_id = f'doc_{i}_parent_{j}'
            parent_doc.metadata['parent_id'] = parent_id
            parent_doc.metadata["parent_content"] = parent_doc.page_content

            # 为父块添加页码和章节元数据
            # 由于 PDF 加载器现在逐页产出 Document，每个 doc 对应一个页码
            # page_start 和 page_end 在此场景下相同
            parent_doc.metadata['page_start'] = src_page
            parent_doc.metadata['page_end'] = src_page
            parent_doc.metadata['chapter'] = src_chapter

            # 切割子块 : [Document]
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])
            for k, sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata['parent_id'] = parent_id
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"

                # 为子块继承页码和章节元数据
                sub_chunk.metadata['page_start'] = src_page
                sub_chunk.metadata['page_end'] = src_page
                sub_chunk.metadata['chapter'] = src_chapter

                child_chunks.append(sub_chunk)

    logger.info(f"[process_documents.process_documents] 子块的数据量：{len(child_chunks)}")

    return child_chunks


if __name__ == '__main__':
    chunks = process_documents(
        directory_path=Config.DATA_DIR,
        parent_chunk_size=Config.PARENT_CHUNK_SIZE,
        child_chunk_size=Config.CHILD_CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    print(len(chunks))
    if chunks:
        print(type(chunks[0]), chunks[0])
