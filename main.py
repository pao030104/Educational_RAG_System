"""
集成问答系统 - 主入口模块
==========================
本模块是整个项目的顶层入口，整合了 MySQL 关键词匹配检索和 RAG 语义检索
两大子系统，提供统一的命令行交互界面。

系统架构:
    ┌─────────────────────────────────────────────────┐
    │              IntegratedQASystem                  │
    │  (main.py - 集成问答系统主控制器)                  │
    ├─────────────────────────────────────────────────┤
    │  ┌──────────────┐  ┌──────────────────────────┐ │
    │  │  mysql_qa    │  │        rag_qa             │ │
    │  │ (关键词检索)  │  │     (语义检索 + LLM)      │ │
    │  │              │  │                           │ │
    │  │ • BM25 匹配  │  │ • 查询分类 (BERT)         │ │
    │  │ • Redis 缓存 │  │ • 策略选择 (LLM)          │ │
    │  │ • MySQL 存储 │  │ • 混合检索 (Milvus)       │ │
    │  │              │  │ • 重排序 (BGE-Reranker)   │ │
    │  └──────────────┘  └──────────────────────────┘ │
    └─────────────────────────────────────────────────┘

查询流程:
    1. 用户输入查询
    2. BM25 关键词匹配 → 如果置信度高，直接返回 MySQL 中的答案
    3. 否则 → RAG 系统进行语义检索，结合 LLM 生成答案
    4. 所有对话历史存储在 MySQL conversations 表中，支持多轮对话

运行方式:
    python main.py
"""

# ---- 导入 MySQL 问答子系统 ----
# MySQLClient: 管理 MySQL 数据库连接和 CRUD 操作
# RedisClient: 管理 Redis 缓存，存储预计算的分词结果和查询缓存
# BM25Search: 基于 BM25 算法的关键词匹配检索引擎
from mysql_qa import MySQLClient, RedisClient, BM25Search

# ---- 导入 RAG 问答子系统 ----
# VectorStore: 基于 Milvus 的向量存储与混合检索引擎
# RAGSystem: RAG 核心逻辑，包含查询分类、策略选择、检索增强生成
from rag_qa import VectorStore, RAGSystem

# ---- 导入基础工具 ----
# logger: 全局日志器，记录系统运行状态和异常信息
# Config: 全局配置管理器，读取 config.ini
from base import logger, Config

# ---- 第三方库导入 ----
# OpenAI: 兼容 OpenAI API 协议的客户端，用于调用阿里云 DashScope LLM 服务
from openai import OpenAI
# time: 用于记录各环节的处理耗时，便于性能分析
import time
# uuid: 生成全局唯一的会话 ID (UUID4)，用于区分不同用户的对话
import uuid
# pymysql: MySQL 数据库驱动，用于类型化的异常处理
import pymysql


class IntegratedQASystem:
    """
    集成问答系统 - 顶层控制器

    整合了以下能力：
        1. BM25 关键词匹配 → 快速精确查找已有问答对
        2. RAG 语义检索  → 理解语义，从知识库中检索相关文档
        3. LLM 答案生成  → 基于检索到的上下文生成自然语言答案
        4. 对话历史管理  → MySQL 持久化存储，支持多轮对话

    属性:
        logger: 日志记录器实例
        config: 全局配置对象
        mysql_client: MySQL 数据库客户端
        redis_client: Redis 缓存客户端
        bm25_search: BM25 关键词检索引擎
        client: OpenAI 兼容的 LLM API 客户端
        vector_store: Milvus 向量存储实例
        rag_system: RAG 检索增强生成系统实例

    使用示例:
        >>> qa = IntegratedQASystem()
        >>> for token, is_complete in qa.query("AI学科学费是多少？", session_id="xxx"):
        ...     print(token, end="")
    """

    def __init__(self):
        """
        初始化集成问答系统，依次建立所有子系统的连接。

        初始化顺序:
            1. 加载配置和日志
            2. 连接 MySQL 数据库
            3. 连接 Redis 缓存
            4. 初始化 BM25 检索引擎
            5. 初始化 LLM 客户端 (DashScope/OpenAI 兼容)
            6. 加载 Milvus 向量存储
            7. 初始化 RAG 系统
            8. 确保 conversations 表存在

        异常处理:
            如果 LLM 客户端或 MySQL 连接失败，将记录错误并抛出异常终止启动。
        """
        # 绑定日志器，用于后续所有操作的日志记录
        self.logger = logger
        # 加载全局配置，包括数据库、LLM、检索等所有参数
        self.config = Config()

        # ---- 启动时配置校验 ----
        config_warnings = self.config.validate()
        for warning in config_warnings:
            self.logger.warning(f"配置警告: {warning}")

        # ---- 初始化数据存储层 ----
        self.mysql_client = MySQLClient()    # 建立 MySQL 数据库连接
        self.redis_client = RedisClient()    # 建立 Redis 缓存连接
        # BM25 检索引擎依赖 MySQL（数据源）和 Redis（缓存），在构造时自动加载数据并构建索引
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

        # ---- 初始化 LLM 客户端 ----
        # 使用 DashScope (阿里云) 作为 LLM 后端，兼容 OpenAI API 协议
        try:
            self.client = OpenAI(
                api_key=self.config.DASHSCOPE_API_KEY,      # API 密钥
                base_url=self.config.DASHSCOPE_BASE_URL,    # API 基础 URL
            )
        except Exception as e:
            self.logger.error(f"OpenAI客户端初始化失败 (请检查API Key和Base URL): {e}")
            raise  # LLM 客户端是系统核心依赖，初始化失败应立即终止

        # ---- 初始化 RAG 子系统 ----
        # VectorStore 加载 Milvus 向量数据库
        self.vector_store = VectorStore()
        # RAGSystem 接收向量存储实例和 LLM 回调函数
        # 传入 call_dashscope (非流式版本)，供内部检索策略使用
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)

        # ---- 确保数据库表结构就绪 ----
        self.init_conversation_table()

    # ==================== 数据库初始化 ====================

    def init_conversation_table(self):
        """
        初始化 MySQL 中的会话历史表 (conversations)。

        表结构:
            - id: 自增主键 (INT AUTO_INCREMENT)
            - session_id: 会话唯一标识 (VARCHAR(36)，用于存储 UUID)
            - question: 用户问题文本 (TEXT)
            - answer: 系统回答文本 (TEXT)
            - timestamp: 对话发生时间 (DATETIME)
            - 索引: session_id 上建立索引，加速按会话查询

        使用 CREATE TABLE IF NOT EXISTS，幂等操作，已存在则跳过。
        """
        try:
            # 执行建表 SQL，IF NOT EXISTS 确保重复执行安全
            self.mysql_client.cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    INDEX idx_session_id (session_id)
                )
            """)
            # 提交数据库事务，确保 DDL 立即生效
            self.mysql_client.connection.commit()
            self.logger.info("初始化历史会话表成功")
        except Exception as e:
            self.logger.error("初始化历史会话表失败:%s" % e)
            raise

    # ==================== LLM 调用接口 ====================

    def call_dashscope(self, prompt):
        """
        调用 DashScope LLM API 生成答案（非流式版本）。

        此方法供 RAGSystem 的内部检索策略使用，因为策略选择、查询分解等
        操作需要返回完整字符串结果。

        参数:
            prompt (str): 完整的提示词文本，已包含上下文和用户问题。

        返回:
            str: LLM 生成的完整答案文本；若出错则返回错误提示字符串。

        注意:
            此方法使用 stream=False，一次性获取完整响应，延迟较高但适合
            需要完整结果才能进行下一步处理的场景。
        """
        try:
            # 构造 API 请求
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,                    # 模型名称，如 qwen3.7-plus
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,     # 30 秒超时，防止请求无限等待
                stream=False,   # 非流式模式，一次性返回完整响应
            )
            # 安全地提取响应内容：逐级检查 choices、message、content 是否有效
            if (completion.choices
                    and completion.choices[0].message
                    and completion.choices[0].message.content):
                return completion.choices[0].message.content
            else:
                self.logger.error("LLM API 调用返回无效响应或空消息")
                return "错误: LLM返回无效响应"
        except Exception as e:
            self.logger.error("错误:调用LLM失败:%s" % e)
            return f"错误:调用LLM失败-{e}"

    def call_dashscope_stream(self, prompt):
        """
        调用 DashScope LLM API 生成答案（流式版本）。

        此方法使用 Server-Sent Events (SSE) 流式传输，逐 token 产出内容，
        适合需要实时展示生成过程的场景，可显著降低用户感知延迟。

        参数:
            prompt (str): 完整的提示词文本。

        Yields:
            str: 逐个产出的文本片段 (token)，调用方通过 for 循环迭代获取。

        使用示例:
            >>> for token in qa.call_dashscope_stream("你好"):
            ...     print(token, end="", flush=True)
        """
        try:
            # 构造流式 API 请求
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
                stream=True,  # 启用流式输出，API 逐块返回内容
            )
            # 逐 token 产出内容
            for chunk in completion:
                # 增量内容位于 chunk.choices[0].delta.content
                # 注意: 非首个 chunk 的 delta.content 可能为空
                # 逐级检查 choices、delta、content 是否有效
                if (chunk.choices
                        and chunk.choices[0].delta
                        and chunk.choices[0].delta.content):
                    content = chunk.choices[0].delta.content
                    yield content  # 逐 token 产出
        except Exception as e:
            self.logger.error("错误:调用LLM失败:%s" % e)
            yield f"错误:调用LLM失败-{e}"

    # ==================== 对话历史管理 ====================

    def _fetch_recent_history(self, session_id: str) -> list:
        """
        从数据库获取指定会话的最近 N 轮对话历史。

        查询策略:
            1. 按时间戳倒序排列，取最近的 LIMIT 条记录
            2. 将数据库行转换为 {"question": ..., "answer": ...} 字典列表
            3. 反转列表，使最早的在前面（时间正序），符合对话上下文构建习惯

        参数:
            session_id (str): 会话唯一标识 UUID。

        返回:
            list[dict]: 对话历史列表，每个元素为 {"question": ..., "answer": ...}；
                       若无历史或查询失败则返回空列表。
        """
        try:
            # 查询指定会话的最近 5 轮对话，按时间降序（最新在前）
            self.mysql_client.cursor.execute(
                """SELECT question,answer FROM conversations
                   WHERE session_id = %s
                   ORDER BY timestamp DESC LIMIT %s""",
                (session_id, 5),
            )
            # 将数据库行转换为易读的字典格式
            history = [
                {"question": row[0], "answer": row[1]}
                for row in self.mysql_client.cursor.fetchall()
            ]
            # 反转列表：数据库返回的是降序（最新在前），反转后为时间正序（从旧到新）
            # 这样构建的对话上下文更符合 LLM 的提示词格式
            return history[::-1]
        except Exception as e:
            self.logger.error("错误:获取最近对话历史失败:%s" % e)
            return []

    def update_session_history(self, session_id, question, answer) -> list:
        """
        更新指定会话的对话历史。

        操作流程:
            1. 插入当前轮的问答对到 conversations 表
            2. 删除超出保留数量的旧记录（仅保留最近 5 轮）
            3. 使用子查询技巧解决 MySQL 不允许在 UPDATE/DELETE 中直接
               SELECT 同一张表的问题

        参数:
            session_id (str): 会话唯一标识。
            question (str): 用户问题文本。
            answer (str): 系统回答文本。

        返回:
            list[dict]: 更新后的最近对话历史。

        异常:
            pymysql.MySQLError: 数据库操作失败时回滚事务并重新抛出。
        """
        try:
            # ---- 步骤1: 插入当前轮对话记录 ----
            # 时间戳使用 MySQL 的 NOW() 函数，保证时间一致性
            self.mysql_client.cursor.execute(
                """
                INSERT INTO conversations(session_id, question, answer, timestamp)
                VALUES(%s, %s, %s, NOW())
                """,
                (session_id, question, answer),
            )
            # 获取更新后的完整历史（此时已包含新插入的记录）
            history = self._fetch_recent_history(session_id)

            # ---- 步骤2: 清理超出保留数量的旧记录 ----
            # MySQL 不支持在子查询中直接引用正在 DELETE 的表，因此使用
            # 嵌套子查询技巧：内层 SELECT 先物化结果，外层再引用
            self.mysql_client.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id=%s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id FROM conversations
                        WHERE session_id=%s
                        ORDER BY timestamp DESC LIMIT %s
                    ) AS sub
                )
            """, (session_id, session_id, 5))

            # ---- 步骤3: 提交事务 ----
            self.mysql_client.connection.commit()
            self.logger.info(f'会话{session_id}历史更新成功')
            return history

        except pymysql.MySQLError as e:
            # 数据库层面错误：回滚事务保证数据一致性，并向上抛出
            self.logger.error("会话历史更新失败:%s" % e)
            self.mysql_client.connection.rollback()
            raise
        except Exception as e:
            # 其他意外错误：同样回滚并抛出
            self.logger.error("更新会话历史意外错误:%s" % e)
            self.mysql_client.connection.rollback()
            raise

    def get_session_history(self, session_id: str) -> list:
        """
        获取指定会话的对话历史（公开接口）。

        参数:
            session_id (str): 会话唯一标识。

        返回:
            list[dict]: 对话历史列表，每个元素含 question 和 answer 字段。
        """
        return self._fetch_recent_history(session_id)

    def clear_session_history(self, session_id: str) -> bool:
        """
        清空指定会话的全部历史记录。

        参数:
            session_id (str): 要清空的会话 ID。

        返回:
            bool: 清空成功返回 True，失败返回 False。

        使用场景:
            用户希望开始全新对话，清除之前的上下文影响。
        """
        try:
            self.mysql_client.cursor.execute(
                "DELETE FROM conversations WHERE session_id = %s", (session_id,)
            )
            self.mysql_client.connection.commit()
            self.logger.info(f"已清空会话历史: {session_id}")
            return True
        except pymysql.MySQLError as e:
            self.logger.error(f"清空会话历史失败: {e}")
            self.mysql_client.connection.rollback()
            return False

    # ==================== 核心查询接口 ====================

    def query(self, query, source_filter=None, session_id=None):
        """
        执行完整的查询流程，依次尝试关键词匹配和语义检索。

        这是系统的核心查询方法，采用两级检索策略:

        ┌─────────────────────────────────────────────────────────┐
        │ 用户查询                                                 │
        │   ↓                                                     │
        │ BM25 关键词检索 ──→ 置信度 ≥ 0.85? ──→ 直接返回答案      │
        │   ↓ 否                                                  │
        │ RAG 语义检索 ──→ 查询分类 → 策略选择 → 检索 → LLM 生成   │
        │   ↓                                                     │
        │ 流式输出答案给用户                                        │
        └─────────────────────────────────────────────────────────┘

        参数:
            query (str): 用户输入的查询文本。
            source_filter (str, optional): 学科类别过滤（如 'ai', 'java'）。
                                           为 None 时不过滤。
            session_id (str, optional): 会话 ID，用于对话历史管理。
                                       为 None 时不记录历史。

        Yields:
            tuple[str, bool]: (文本片段, 是否完成)
                - is_complete=True: 当前为完整答案模式，yield 一次即完成
                - is_complete=False: 预留的流式模式（当前版本不使用）

        使用示例:
            >>> for token, done in qa.query("什么是机器学习？", session_id="abc"):
            ...     if done:
            ...         print("回答完毕")
        """
        # 记录查询开始时间，用于计算总耗时
        start_time = time.time()
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id})")

        # 如果提供了会话 ID，获取最近的对话历史（暂未在此版本中用于上下文构建）
        history = self.get_session_history(session_id) if session_id else []

        # ---- 第1级: BM25 关键词匹配检索 ----
        # search 方法返回 (answer, need_rag) 元组
        # - answer 有值: 在知识库中找到高置信度匹配
        # - need_rag=True: BM25 匹配失败或置信度不足，需要 RAG 进一步处理
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)

        if answer:
            # ---- 分支A: BM25 命中，直接返回精确匹配答案 ----
            self.logger.info(f"从MySQL获取答案: '{answer}' (会话ID: {session_id})")
            if session_id:
                # 有会话 ID 时记录本次对话历史
                self.update_session_history(session_id, query, answer)
            processing_time = time.time() - start_time
            self.logger.info(f"处理时间: {processing_time:.2f} 秒")
            # 产出答案并标记完成
            yield answer, True

        elif need_rag:
            # ---- 分支B: BM25 未命中，启动 RAG 语义检索流程 ----
            self.logger.info("无可靠MySQL答案,使用RAG系统处理查询")
            # RAGSystem.generate_answer 内部执行:
            #   1) BERT 查询分类（通用知识 vs 专业咨询）
            #   2) LLM 策略选择（直接/子查询/回溯/HyDE）
            #   3) Milvus 混合检索 + BGE-Reranker 重排序
            #   4) LLM 基于检索到的上下文生成最终答案
            answer = self.rag_system.generate_answer(query, source_filter=source_filter)
            if session_id:
                self.update_session_history(session_id, query, answer)
            processing_time = time.time() - start_time
            self.logger.info(f'处理完成,用时:{processing_time:.2f}秒')
            # 产出完整答案并标记完成
            yield answer, True

        else:
            # ---- 分支C: 两级检索均无结果 ----
            self.logger.info(f"未找到答案")
            processing_time = time.time() - start_time
            self.logger.info(f'处理完成,用时:{processing_time:.2f}秒')
            yield "未找到答案", True


# ==================== 命令行入口 ====================

def main():
    """
    主函数 - 提供命令行交互式问答界面。

    功能:
        1. 初始化 IntegratedQASystem（完成所有子系统的连接）
        2. 生成唯一的会话 ID（UUID4）
        3. 进入 REPL 循环：读取用户输入 → 执行查询 → 输出答案
        4. 支持学科类别过滤
        5. 输入 'exit' 退出程序

    会话管理:
        - 每次启动程序生成新的会话 ID
        - 同一会话内的多轮对话被记录在 MySQL 中
        - 退出后自动关闭 MySQL 连接
    """
    # 实例化集成问答系统，完成所有子系统的初始化和连接
    qa_system = IntegratedQASystem()

    # 生成全局唯一的会话 ID (UUID4 = 基于随机数的 UUID)
    # 格式示例: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
    session_id = str(uuid.uuid4())

    # ---- 打印欢迎信息和操作指南 ----
    print("\n欢迎使用集成问答系统！")
    print(f"会话ID: {session_id}")
    print(f"支持的学科类别：{qa_system.config.VALID_SOURCES}")
    print("输入查询进行问答，输入 'exit' 退出。")

    try:
        # ---- REPL (Read-Eval-Print Loop) 主循环 ----
        while True:
            # 读取用户输入，strip 去除首尾空白
            query = input("请输入查询:").strip()

            # 检查退出条件
            if query.lower() == "exit":
                logger.info("退出集成问答系统")
                print("感谢使用集成问答系统！再见！")
                break

            # 读取学科类别过滤（可选）
            # 用户可直接回车跳过，表示不过滤
            source_filter = input(
                f"请输入学科类别 ({'/'.join(qa_system.config.VALID_SOURCES)}) (直接回车默认不过滤): "
            ).strip()

            # 验证输入的学科类别是否有效
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                logger.warning(f"无效的学科类别 '{source_filter}'，将不过滤")
                source_filter = None  # 无效输入视为不过滤

            # 输出 "答案:" 前缀，准备接收流式输出
            print("\n答案:", end="", flush=True)
            answer = ""

            # 执行查询，逐片段接收答案
            # query 方法是生成器，yield (token, is_complete) 元组
            for token, is_complete in qa_system.query(
                query, source_filter=source_filter, session_id=session_id
            ):
                if token:
                    answer += token                     # 拼接完整答案
                    print(token, end="", flush=True)    # 逐片段打印，flush 保证实时显示
                if is_complete:
                    print()  # 答案完整输出后换行
                    break

            # ---- 显示最近对话历史 ----
            history = qa_system.get_session_history(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")

    except KeyboardInterrupt:
        # 用户按下 Ctrl+C，优雅退出
        logger.info("用户通过 Ctrl+C 退出系统")
        print("\n\n感谢使用集成问答系统！再见！")
    except Exception as e:
        # 捕获所有未预期的异常，记录日志并提示用户
        logger.error(f'系统错误:{e}')
        print(f'发生错误:{e}')
    finally:
        # 无论正常退出还是异常退出，确保关闭 MySQL 连接释放资源
        qa_system.mysql_client.close()


# Python 标准入口：当直接执行此文件时运行 main()，被 import 时不执行
if __name__ == '__main__':
    main()
