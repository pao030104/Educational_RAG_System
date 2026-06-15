"""
edu_document_loaders 包 - 教育文档加载器模块
=============================================
提供面向教育场景的多格式文档加载器，支持 OCR 识别图片中的文字。

导出组件:
    - OCRPDFLoader:  PDF 文档加载器（含嵌入式图片 OCR）
    - OCRDOCLoader:  Word (.docx) 文档加载器（含嵌入式图片 OCR）
    - OCRPPTLoader:  PowerPoint (.ppt/.pptx) 加载器（含图片 OCR）
    - OCRIMGLoader:  图片文件 OCR 加载器 (.jpg/.png)

核心能力:
    1. 文本提取: 提取文档中的原生文本内容（段落、表格、文本框等）
    2. 图片 OCR: 对文档中嵌入的图片进行 OCR 文字识别
    3. 统一接口: 所有加载器遵循 LangChain BaseLoader 接口规范
    4. 进度追踪: 使用 tqdm 显示处理进度

OCR 引擎:
    使用 RapidOCR（支持 PaddlePaddle 和 ONNX Runtime 两种后端）:
        - GPU 可用时: rapidocr_paddle (PaddlePaddle GPU 加速)
        - 仅 CPU 时:   rapidocr_onnxruntime (ONNX Runtime 优化)

文档处理流程:
    文件 → 解析结构 → 提取文本 + 识别图片文字 → 拼接输出 → LangChain Document
"""

import sys
import os

# 将当前目录加入 Python 路径
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)

# ---- 导出所有文档加载器 ----
from edu_docloader import *    # OCRDOCLoader - Word 文档加载器
from edu_pptloader import *    # OCRPPTLoader - PowerPoint 加载器
from edu_imgloader import *    # OCRIMGLoader - 图片 OCR 加载器
from edu_pdfloader import *    # OCRPDFLoader - PDF 文档加载器
