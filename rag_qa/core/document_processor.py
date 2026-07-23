"""
文档加载与处理模块
===================
负责从多种文件格式加载文档，并进行父子块分层切分 (Parent-Child Chunking)。

支持的文件类型:
    ┌──────────────┬─────────────────────────────────────┐
    │ 扩展名       │ 加载器                              │
    ├──────────────┼─────────────────────────────────────┤
    │ .txt         │ TextLoader (UTF-8)                   │
    │ .pdf         │ OCRPDFLoader (含图片OCR)             │
    │ .docx        │ OCRDOCLoader (含图片OCR)             │
    │ .ppt/.pptx   │ OCRPPTLoader (含图片OCR)             │
    │ .jpg/.png    │ OCRIMGLoader (图片OCR)               │
    │ .md          │ TextLoader           │
    └──────────────┴─────────────────────────────────────┘

父子块 (Parent-Child Chunk) 策略说明:

    目的: 解决大块上下文和小块检索精度之间的矛盾。

    ┌─────────────────────────────────────────────┐
    │ 父块 (1200 字符)                            │
    │ ┌─────────────┬─────────────┬─────────────┐ │
    │ │ 子块1 (300) │ 子块2 (300) │ 子块3 (300) │ │
    │ │   检索用    │   检索用    │   检索用    │ │
    │ └─────────────┴─────────────┴─────────────┘ │
    │           ↑ 返回父块作为 LLM 上下文 ↑        │
    └─────────────────────────────────────────────┘

    1. 子块 (300字符): 用于向量化和检索 ── 小块检索精度高
    2. 父块 (1200字符): 包含完整上下文，返回给 LLM ── 大块上下文充足

切分器选择:
    - Markdown 文件: MarkdownTextSplitter（保留标题层级结构）
    - 其他文件: ChineseRecursiveTextSplitter（中文递归分割，按标点智能切分）
"""

import os  # 文件和目录操作

# ---- LangChain 文档加载器 ----
from langchain_community.document_loaders import TextLoader             # 纯文本加载器

# ---- 文本切分器 ----
from langchain.text_splitter import MarkdownTextSplitter                # Markdown 专用切分器

from datetime import datetime                                           # 时间戳生成

# ---- 项目自定义组件 ----
# 中文递归文本分割器（支持按句号、感叹号、分号等中文标点切分）
from ..edu_text_spliter import AliTextSplitter, ChineseRecursiveTextSplitter
# 自定义文档加载器（支持 OCR 识别 PDF/DOCX/PPT 中的图片文字）
from ..edu_document_loaders import (
    OCRPDFLoader,      # PDF 加载器
    OCRDOCLoader,      # Word 文档加载器
    OCRPPTLoader,      # PowerPoint 加载器
    OCRIMGLoader       # 图片 OCR 加载器
)
from base import logger, Config  # 日志和配置

conf = Config()  # 全局配置实例


# ==================== 文件类型 → 加载器映射表 ====================
# 根据文件扩展名选择对应的 LangChain 兼容加载器
document_loaders = {
    ".txt":  TextLoader,                # 纯文本文件
    ".pdf":  OCRPDFLoader,              # PDF 文档（含 OCR）
    ".docx": OCRDOCLoader,              # Word 文档（含 OCR）
    ".ppt":  OCRPPTLoader,              # PowerPoint 演示文稿（含 OCR）
    ".pptx": OCRPPTLoader,              # PowerPoint 新版格式
    ".jpg":  OCRIMGLoader,              # JPEG 图片（OCR 识别）
    ".png":  OCRIMGLoader,              # PNG 图片（OCR 识别）
    ".md":   TextLoader                  # Markdown 文档（用 TextLoader 替代，避免 spaCy 网络依赖）
}


# ==================== 文档加载 ====================

def load_documents_from_directory(directory_path):
    """
    从指定目录递归加载所有支持的文档文件，并添加元数据。

    处理流程:
        1. 遍历目录及所有子目录
        2. 根据文件扩展名选择对应的加载器
        3. 加载文档内容（PDF/DOCX/PPT 包含图片 OCR）
        4. 为每个文档添加元数据: source (学科类别), file_path, timestamp

    目录命名约定:
        目录名格式为 "{source}_data"，例如:
            - "ai_data" → source = "ai"
            - "java_data" → source = "java"
        系统会自动从目录名提取学科类别作为 source 元数据。

    参数:
        directory_path (str): 待加载的文档目录路径。

    返回:
        list[Document]: 所有加载成功的 LangChain Document 对象列表。
                       不支持的格式会被跳过并记录警告。

    示例:
        >>> docs = load_documents_from_directory("./data/ai_data")
        >>> print(f"加载了 {len(docs)} 个文档")
    """
    documents = []  # 累积所有加载的文档

    # ---- 获取支持的文件扩展名集合 ----
    supported_extensions = document_loaders.keys()

    # ---- 从目录名提取学科类别 ----
    # os.path.basename 获取目录最后一级名称，然后去掉 "_data" 后缀
    # 例如: "./data/ai_data" → "ai_data" → "ai"
    # 使用 removesuffix 确保只移除末尾的 "_data"，避免误替换中间的 "_data"
    base_dirname = os.path.basename(directory_path.rstrip("/"))
    source = base_dirname.removesuffix("_data") if base_dirname.endswith("_data") else base_dirname

    # ---- 递归遍历目录 ----
    # os.walk 递归遍历所有子目录，返回 (root, dirs, files)
    for root, _, files in os.walk(directory_path):
        for file in files:
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            # 获取文件扩展名并转为小写（保证大小写不敏感匹配）
            file_extension = os.path.splitext(file_path)[1].lower()

            # ---- 检查文件类型是否支持 ----
            if file_extension in supported_extensions:
                try:
                    # 根据扩展名获取对应的加载器类
                    loader_class = document_loaders[file_extension]

                    # ---- 实例化加载器 ----
                    # 不同加载器的构造参数不同
                    if file_extension == ".txt":
                        # 文本文件需要指定 UTF-8 编码，否则可能乱码
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)

                    # 加载文档内容
                    loaded_docs = loader.load()

                    # ---- 为每个文档添加元数据 ----
                    for doc in loaded_docs:
                        doc.metadata["source"] = source                 # 学科类别
                        doc.metadata["file_path"] = file_path            # 源文件路径
                        doc.metadata["timestamp"] = datetime.now().isoformat()  # 处理时间戳

                    # 添加到总文档列表
                    documents.extend(loaded_docs)
                    logger.info(f"成功加载文件: {file_path}")

                except Exception as e:
                    # 单个文件加载失败不影响其他文件
                    logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            else:
                # 不支持的文件类型，记录警告
                logger.warning(f"不支持的文件类型: {file_path}")

    return documents


# ==================== 文档分层切分 ====================

def process_documents(
    directory_path,
    parent_chunk_size=conf.PARENT_CHUNK_SIZE,
    child_chunk_size=conf.CHILD_CHUNK_SIZE,
    chunk_overlap=conf.CHUNK_OVERLAP
):
    """
    加载文档并进行父子块分层切分。

    这是文档进入向量数据库前的完整预处理管道:

    ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
    │ 加载文档      │───→│ 父块切分 (1200)   │───→│ 子块切分 (300)│
    │ (多格式支持)  │    │ (保留大段上下文)  │    │ (用于精确检索)│
    └──────────────┘    └──────────────────┘    └──────────────┘
                                                        │
                                                        ▼
                                                ┌──────────────┐
                                                │ 子块关联父块  │
                                                │ (parent_id   │
                                                │  + parent_    │
                                                │  content)     │
                                                └──────────────┘

    参数:
        directory_path (str):   待处理的文档目录路径。
        parent_chunk_size (int): 父块大小（字符数），默认 1200。
        child_chunk_size (int):  子块大小（字符数），默认 300。
        chunk_overlap (int):    相邻块之间的重叠字符数，默认 50。
                                重叠可以避免信息在切分边界处丢失。

    返回:
        list[Document]: 所有子块组成的 Document 列表，每个子块包含:
                        - page_content: 子块文本
                        - metadata.parent_id: 关联的父块 ID
                        - metadata.parent_content: 关联的父块完整内容

    使用示例:
        >>> chunks = process_documents(
        ...     "./data/ai_data",
        ...     parent_chunk_size=1200,
        ...     child_chunk_size=300,
        ...     chunk_overlap=50
        ... )
        >>> print(f"生成了 {len(chunks)} 个子块")
    """
    # ---- 步骤1: 加载所有文档 ----
    documents = load_documents_from_directory(directory_path)
    logger.info(f"加载的文档数量: {len(documents)}")

    # ---- 步骤2: 初始化切分器 ----
    # 通用中文切分器（适用于 txt, pdf, docx, ppt 等）
    parent_splitter = ChineseRecursiveTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=chunk_overlap
    )
    child_splitter = ChineseRecursiveTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Markdown 专用切分器（保留 Markdown 标题层级和代码块结构）
    markdown_parent_splitter = MarkdownTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=chunk_overlap
    )
    markdown_child_splitter = MarkdownTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap
    )

    # ---- 步骤3: 逐文档切分 ----
    child_chunks = []  # 累积所有子块

    for i, doc in enumerate(documents):
        # ---- 确定当前文档应使用的切分器 ----
        file_extension = os.path.splitext(
            doc.metadata.get("file_path", "")
        )[1].lower()
        is_markdown = (file_extension == ".md")

        # 根据是否为 Markdown 选择对应的切分器
        parent_splitter_to_use = (
            markdown_parent_splitter if is_markdown else parent_splitter
        )
        child_splitter_to_use = (
            markdown_child_splitter if is_markdown else child_splitter
        )

        logger.info(
            f"处理文档: {doc.metadata['file_path']}, "
            f"使用切分器: {'Markdown' if is_markdown else 'ChineseRecursive'}"
        )

        # ---- 切分为父块 ----
        # split_documents 接收 Document 列表，返回切分后的 Document 列表
        parent_docs = parent_splitter_to_use.split_documents([doc])

        # ---- 每个父块再切分为子块 ----
        for j, parent_doc in enumerate(parent_docs):
            # 为父块生成唯一标识符，格式: doc_{文档序号}_parent_{父块序号}
            parent_id = f"doc_{i}_parent_{j}"
            parent_doc.metadata["parent_id"] = parent_id
            # 将父块的完整内容保存在元数据中（子块会继承）
            parent_doc.metadata["parent_content"] = parent_doc.page_content

            # 将父块进一步切分为子块
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])

            # ---- 为每个子块添加父块关联信息 ----
            for k, sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata["parent_id"] = parent_id                 # 关联父块ID
                sub_chunk.metadata["parent_content"] = parent_doc.page_content  # 父块完整内容
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"         # 子块唯一标识
                child_chunks.append(sub_chunk)

    logger.info(f"子块数量: {len(child_chunks)}")
    return child_chunks


# ==================== 直接运行入口 ====================
if __name__ == '__main__':
    # 测试文档处理功能
    # 注意：此路径需要根据实际环境修改
    import sys
    test_dir = sys.argv[1] if len(sys.argv) > 1 else './data/ai_data'
    chunks = process_documents(
        test_dir,
        conf.PARENT_CHUNK_SIZE,
        conf.CHILD_CHUNK_SIZE,
        conf.CHUNK_OVERLAP,
    )
    print(chunks)
