"""
rag_qa 包 - RAG 检索增强生成问答子系统
=======================================
基于 Milvus 向量数据库和 LLM 的语义检索问答系统。

系统组成:
    - core/vector_store.py:          Milvus 向量存储与混合检索引擎
    - core/rag_system.py:            RAG 核心控制链路（分类→策略→检索→生成）
    - core/document_processor.py:    多格式文档加载与父子块切分
    - core/prompts.py:               LLM Prompt 模板管理
    - core/query_classifier.py:      BERT 查询意图分类器
    - core/strategy_selector.py:     LLM 检索策略选择器
    - edu_text_spliter/:             中文文本分割器
    - edu_document_loaders/:         多格式文档加载器（含 OCR）

导出组件:
    - VectorStore: 向量存储与混合检索
    - RAGSystem:   RAG 检索增强生成主控

注意:
    此 __init__.py 负责设置 Python 路径，确保包内模块可以互相导入。
    导入 VectorStore 和 RAGSystem 两个最核心的类供外部使用。
"""

import os
import sys

# ---- 设置 Python 模块搜索路径 ----
# 将 rag_qa 目录加入 sys.path，确保 core、edu_text_spliter 等子包可以被正确导入
current_dir = os.path.abspath(__file__)   # __init__.py 的绝对路径
rag_qa_path = os.path.dirname(current_dir) # rag_qa 目录
sys.path.insert(0, rag_qa_path)

# ---- 导出核心组件 ----
from core.vector_store import VectorStore   # Milvus 向量存储与检索
from core.rag_system import RAGSystem       # RAG 核心系统
# 如需使用带流式输出的版本，替换为:
# from core.new_rag_system import RAGSystem
