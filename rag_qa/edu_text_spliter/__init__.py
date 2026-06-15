"""
edu_text_spliter 包 - 教育文本分割器模块
=========================================
提供面向中文教育文本的专用分割器，支持递归分割和基于深度学习的语义分割。

导出组件:
    - AliTextSplitter:              阿里达摩院文档语义分割器（基于 BERT 模型）
    - ChineseRecursiveTextSplitter: 中文递归文本分割器（支持中文标点智能切分）

使用场景:
    - ChineseRecursiveTextSplitter: 通用中文文档切分，按句号、感叹号、分号等标点递归分割
    - AliTextSplitter:             需要语义级文档切分的场景，基于 nlp_bert_document-segmentation
"""

import sys
import os

# 将当前目录加入 Python 路径，确保子模块可以互相导入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# ---- 导出两个文本分割器 ----
from edu_model_text_spliter import *                    # AliTextSplitter (BERT 语义分割)
from edu_chinese_recursive_text_splitter import *       # ChineseRecursiveTextSplitter (递归分割)
