"""
RAG 问答子系统 - 独立运行入口
==============================
提供 rag_qa 子系统的独立命令行界面。
支持两种运行模式：数据处理模式 和 交互式查询模式。

运行方式:
    # 交互式查询模式（默认）
    python -m rag_qa.rag_main
    或
    python rag_qa/rag_main.py

    # 数据处理模式（将文档导入向量数据库）
    python rag_qa/rag_main.py --data_processing
    python rag_qa/rag_main.py --data_processing --data_dir ./data

模式说明:
    1. 数据处理模式 (--data_processing):
       扫描指定目录下的文档，执行预处理和向量化，存入 Milvus。
       适用于系统初始化或知识库更新。

    2. 交互式查询模式（默认）:
       启动命令行问答界面，逐条回答用户问题。
       适用场景: 系统演示、功能测试、手动问答。
"""

# -*-coding:utf-8-*-
import os
import sys

# ---- 设置 Python 模块搜索路径 ----
rag_qa_path = os.path.dirname(os.path.abspath(__file__))       # rag_qa/
sys.path.insert(0, rag_qa_path)
core_path = os.path.join(rag_qa_path, 'core')                  # rag_qa/core/
sys.path.insert(0, core_path)
project_root = os.path.dirname(rag_qa_path)                     # 项目根目录
sys.path.insert(0, project_root)

from base import Config, logger                                # 配置和日志
from rag_qa.core.document_processor import process_documents           # 文档处理函数
from rag_qa.core.vector_store import VectorStore                       # 向量存储
from rag_qa.core.rag_system import RAGSystem                           # RAG 核心系统
from openai import OpenAI                                       # LLM API 客户端

conf = Config()  # 全局配置实例


def main(query_mode=True, directory_path="data"):
    """
    RAG 子系统主函数，支持数据处理和交互式查询两种模式。

    参数:
        query_mode (bool): True 为查询模式，False 为数据处理模式。
        directory_path (str): 数据目录路径。查询模式下不使用。

    数据处理模式流程:
        1. 连接 Milvus
        2. 遍历 VALID_SOURCES（如 ai, java, bigdata）
        3. 加载 {source}_data 目录下的所有文档
        4. 父子块切分 → 向量化 → 存入 Milvus

    查询模式流程:
        1. 连接 Milvus 和 LLM
        2. 初始化 RAGSystem
        3. 进入 REPL 循环
        4. 每次查询: BERT分类 → 策略选择 → 检索 → LLM生成答案

    使用示例:
        >>> main(query_mode=False, directory_path="./data")  # 数据处理
        >>> main(query_mode=True)                            # 交互式查询
    """
    # ==================== LLM 客户端初始化 ====================
    try:
        # 初始化 OpenAI 兼容客户端（连接 DashScope API）
        client = OpenAI(
            api_key=conf.DASHSCOPE_API_KEY,
            base_url=conf.DASHSCOPE_BASE_URL
        )
    except Exception as e:
        logger.error(f"初始化 OpenAI 客户端失败 (请检查 API Key 和 Base URL): {e}")
        if query_mode:
            # 查询模式必须有 LLM，否则无法回答
            print("错误：无法初始化语言模型客户端，无法进入查询模式。")
            return
        client = None  # 数据处理模式可能不需要 LLM

    def call_dashscope(prompt):
        """
        LLM 调用函数（非流式）。

        参数:
            prompt (str): 完整的提示词文本。

        返回:
            str: LLM 生成的答复；若客户端不可用或调用失败，返回错误提示。
        """
        if not client:
            logger.error("LLM 客户端未初始化，无法调用 call_dashscope")
            return f"错误: LLM客户端不可用"

        try:
            completion = client.chat.completions.create(
                model=conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手"},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
                # 注意: 此处未设置 stream，默认非流式
            )
            # 安全提取响应
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            else:
                logger.error("LLM API 调用返回无效响应或空消息")
                return "错误: LLM返回无效响应"
        except Exception as e:
            logger.error(f"LLM API (call_dashscope) 调用失败: {e}")
            return f"错误: 调用LLM失败 - {e}"

    # ==================== 向量存储初始化 ====================
    try:
        vector_store = VectorStore(
            collection_name=conf.MILVUS_COLLECTION_NAME,
            host=conf.MILVUS_HOST,
            port=conf.MILVUS_PORT,
            database=conf.MILVUS_DATABASE_NAME,
        )
    except Exception as e:
        logger.error(f"初始化 VectorStore 失败 (请检查 Milvus 连接配置): {e}")
        print("错误：无法连接到向量数据库，程序无法继续。")
        return

    # ==================== 模式分派 ====================
    if not query_mode:
        # ---- 数据处理模式 ----
        logger.info("进入数据处理模式....")
        total_chunks_added = 0  # 累计添加的子块总数

        # ---- 收集待处理的目录列表 ----
        dirs_to_process = []
        for source_dir in conf.VALID_SOURCES:
            # 约定: 每个学科的数据放在 {source}_data 目录下
            dir_path = os.path.join(directory_path, f"{source_dir}_data")
            if os.path.exists(dir_path):
                dirs_to_process.append((source_dir, dir_path))
            else:
                logger.warning(f"目录 {dir_path} 不存在，跳过处理")

        # ---- 兼容处理 ----
        # 如果没有找到 {source}_data 子目录，则直接处理 directory_path
        if not dirs_to_process and os.path.exists(directory_path):
            dir_basename = os.path.basename(directory_path.rstrip("/"))
            # 尝试从目录名提取学科类别
            if dir_basename.endswith("_data"):
                source = dir_basename[:-5]  # 去掉 _data 后缀
            else:
                source = dir_basename       # 直接使用目录名
            dirs_to_process.append((source, directory_path))

        # ---- 逐目录处理 ----
        for source, dir_path in dirs_to_process:
            logger.info(f"开始处理目录:{dir_path}")
            try:
                # 加载文档 → 父子块切分 → 返回子块列表
                chunks = process_documents(
                    dir_path,
                    conf.PARENT_CHUNK_SIZE,   # 父块大小 (默认1200)
                    conf.CHILD_CHUNK_SIZE,    # 子块大小 (默认300)
                    conf.CHUNK_OVERLAP,       # 块重叠大小 (默认50)
                )
                if chunks:
                    # 向量化并存入 Milvus
                    vector_store.add_document(chunks)
                    total_chunks_added += len(chunks)
                    logger.info(
                        f"成功处理目录 {dir_path}，添加了 {len(chunks)} 个文档块"
                    )
                else:
                    logger.info(f"目录 {dir_path} 未发现有效文档或处理结果为空")
            except Exception as e:
                logger.error(f"处理目录{dir_path}时出错:{e}")

        logger.info(f"数据处理完成，共添加 {total_chunks_added} 个文档块")

    else:
        # ---- 交互式查询模式 ----
        if not client:
            print("错误：查询模式需要语言模型客户端，但初始化失败。")
            return

        logger.info("进入交互式查询模式...")

        # 初始化 RAG 系统
        try:
            rag_system = RAGSystem(vector_store, call_dashscope)
        except Exception as e:
            logger.error(f"初始化 RAGSystem 失败: {e}")
            print("错误：无法初始化 RAG 系统，无法进入查询模式。")
            return

        valid_sources = conf.VALID_SOURCES  # 有效学科类别列表

        # ---- 欢迎信息 ----
        print("\n欢迎使用 EduRAG 交互式查询系统！")
        print(f"支持的学科类别：{valid_sources}")
        print("输入您的问题，或输入 'exit' 退出。")

        # ---- REPL 主循环 ----
        while True:
            # 读取用户查询
            query = input("\n请输入您的问题:")

            # 检查退出条件
            if query.lower() == "exit":
                logger.info("退出 RAG 系统")
                print("感谢使用 EduRAG ！再见！")
                break

            # 读取学科类别过滤
            source_filter_input = input(
                f"请输入学科类别 ({'/'.join(valid_sources)}) (直接回车默认不过滤)："
            ).strip()

            source_filter = None  # 默认不过滤
            if source_filter_input:
                if source_filter_input in valid_sources:
                    source_filter = source_filter_input
                    logger.info(f"过滤学科类别为：{source_filter}")
                else:
                    logger.warning(
                        f"无效的学科类别 '{source_filter_input}'，将不过滤"
                    )
                    print(
                        f"提示：输入的学科 '{source_filter_input}' 无效，将不过滤。"
                    )

            # ---- 执行 RAG 查询 ----
            try:
                print("正在生成答案，请稍候...")
                # RAGSystem.generate_answer 内部完成:
                #   分类 → 策略选择 → 检索 → LLM 生成
                answer, _, _ = rag_system.generate_answer(
                    query, source_filter=source_filter
                )
                # 格式化输出
                print("-" * 30)
                print(f"问题: {query}")
                print(f"回答: {answer}")
                print("-" * 30)
            except Exception as e:
                logger.error(f"处理查询 '{query}' 时失败: {str(e)}")
                print(
                    f"抱歉，处理您的问题时遇到了错误，请稍后重试或联系管理员。\n"
                )


# ==================== 命令行入口 ====================
if __name__ == "__main__":
    import argparse

    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(
        description="EduRAG System Main Entry Point"
    )
    # --data_processing: 切换到数据处理模式
    parser.add_argument(
        '--data_processing',
        action='store_true',
        help='Run in data processing mode instead of query mode.'
    )
    # --data_dir: 指定数据目录路径
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Path to the data directory.'
    )
    args = parser.parse_args()

    # 调用 main 函数
    # --data_processing 存在 → query_mode=False → 数据处理模式
    # 否则 → query_mode=True → 交互式查询模式
    main(query_mode=(not args.data_processing), directory_path=args.data_dir)
