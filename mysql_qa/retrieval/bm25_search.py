"""
BM25 关键词检索引擎模块
========================
基于 BM25 算法的文本检索系统，用于在已有问答对知识库中进行关键词匹配。

BM25 (Best Matching 25) 算法简介:
    BM25 是 TF-IDF 的改进版本，通过以下方式克服了 TF-IDF 的不足：
    1. 词频饱和处理：词频达到一定程度后不再线性增长
    2. 文档长度归一化：长文档不会因词多而占优势
    3. 可调参数 k1 和 b：分别控制词频饱和度和长度归一化程度

本系统中的检索流程:
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │ 用户查询      │───→│ jieba 分词    │───→│ BM25 打分    │
    └──────────────┘    └──────────────┘    └──────────────┘
                                                   │
                                                   ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │ 返回答案      │←───│ MySQL 查答案  │←───│ Softmax 阈值  │
    └──────────────┘    └──────────────┘    └──────────────┘

缓存策略:
    - 首次加载: MySQL → 分词 → 存 Redis + 构建 BM25 索引
    - 后续加载: Redis 直接读取（跳过 MySQL 和分词）

阈值设计:
    使用 Softmax 归一化后的分数，默认阈值为 0.85。
    Softmax 将所有候选分数映射到 [0, 1] 区间且和为 1，
    使得阈值具有更直观的解释性。
"""

from rank_bm25 import BM25Okapi  # BM25 算法实现，接收分词后的文档列表
import numpy as np               # 数值计算库，用于 Softmax 等数学运算
from ..utils.preprocess import preprocess_text  # 中文文本预处理（分词）
from base import logger          # 全局日志器


class BM25Search:
    """
    BM25 关键词检索器

    封装了 BM25 索引的构建、数据加载和查询功能。
    自动管理 Redis 缓存和 MySQL 数据源之间的数据同步。

    属性:
        logger:             日志记录器
        redis_client:       Redis 缓存客户端
        mysql_client:       MySQL 数据库客户端
        bm25:               BM25Okapi 模型实例（None 表示未初始化）
        original_questions: 原始问题文本列表（统一为 list[str]）
        questions:          分词后的问题列表（list[list[str]]）

    使用示例:
        >>> redis_cli = RedisClient()
        >>> mysql_cli = MySQLClient()
        >>> bm25 = BM25Search(redis_cli, mysql_cli)
        >>> answer, need_rag = bm25.search("什么是人工智能?", threshold=0.85)
        >>> if answer:
        ...     print(answer)
        ... else:
        ...     print("需要RAG处理" if need_rag else "未找到答案")
    """

    def __init__(self, redis_client, mysql_client):
        """
        初始化 BM25 检索器，加载数据并构建索引。

        参数:
            redis_client (RedisClient): 已初始化的 Redis 客户端。
            mysql_client (MySQLClient): 已初始化的 MySQL 客户端。
        """
        self.logger = logger
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        # BM25 模型实例，在 _load_data 中初始化
        self.bm25 = None
        # 原始问题文本列表，备查时将索引映射回问题/答案
        self.original_questions = None
        # 加载数据并构建 BM25 索引
        self._load_data()

    def _load_data(self):
        """
        加载问答数据并构建 BM25 索引。

        数据加载优先级:
            1. 优先从 Redis 缓存读取（快速启动）
            2. 若缓存未命中，从 MySQL 加载 → 分词 → 存入 Redis 缓存

        缓存键:
            - qa_original_questions: 原始问题文本列表 (JSON 序列化)
            - qa_tokenized_questions: 分词结果列表 (JSON 序列化)

        异常安全:
            若 MySQL 中无数据，会将 original_questions、questions 设为空列表，
            bm25 设为 None，并在 search 方法中进行守卫检查，避免崩溃。
        """
        # ---- 定义 Redis 缓存键名 ----
        original_key = 'qa_original_questions'
        tokenized_key = 'qa_tokenized_questions'

        # ---- 步骤1: 尝试从 Redis 缓存读取 ----
        self.original_questions = self.redis_client.get_data(original_key)
        tokenized_questions = self.redis_client.get_data(tokenized_key)

        # ---- 步骤2: 缓存未命中时从 MySQL 加载 ----
        if not self.original_questions or not tokenized_questions:
            # 从 MySQL 获取所有问题（返回 list[tuple]）
            raw_questions = self.mysql_client.fetch_questions()

            if not raw_questions:
                # MySQL 中也没有数据，无法初始化索引
                self.logger.error('未加载到问题')
                self.original_questions = []  # 空列表代替 None
                self.questions = []           # 空列表代替未定义
                self.bm25 = None              # 显式标记为未初始化
                return

            # 将 MySQL 返回的元组列表统一转换为字符串列表
            # raw_questions 格式: [("q1",), ("q2",), ...]  →  ["q1", "q2", ...]
            self.original_questions = [q[0] for q in raw_questions]

            # 对每个问题进行中文分词，构建 Bag-of-Words 表示
            tokenized_questions = [
                preprocess_text(q) for q in self.original_questions
            ]

            # ---- 存入 Redis 缓存，加速下次启动 ----
            # 存储原始问题（字符串列表）
            self.redis_client.set_data(original_key, self.original_questions)
            # 存储分词结果（列表的列表）
            self.redis_client.set_data(tokenized_key, tokenized_questions)

        # ---- 步骤3: 构建 BM25 索引 ----
        self.questions = tokenized_questions
        # BM25Okapi 在构造时自动计算所有文档的 IDF 和平均长度等统计量
        self.bm25 = BM25Okapi(self.questions)
        self.logger.info('BM25模型初始化完成')

    def _softmax(self, scores):
        """
        对 BM25 原始分数应用 Softmax 归一化。

        Softmax 公式:
            softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))

        减去最大值 (max(x)) 是为了数值稳定性：
            - 避免 exp 对大值溢出
            - 不改变最终结果的相对关系

        参数:
            scores (np.ndarray): BM25 原始分数数组。

        返回:
            np.ndarray: Softmax 归一化后的概率分布，总和为 1。
        """
        # 减去最大值以保证数值稳定性
        exp_scores = np.exp(scores - np.max(scores))
        # 除以总和得到概率分布
        return exp_scores / np.sum(exp_scores)

    def search(self, query, threshold=0.85):
        """
        执行 BM25 关键词检索。

        检索流程:
            1. 验证输入有效性
            2. 检查 BM25 索引是否就绪
            3. 查询 Redis 缓存
            4. 对查询文本分词
            5. 计算与所有已知问题的 BM25 分数
            6. Softmax 归一化
            7. 取最高分，若超过阈值则从 MySQL 获取对应答案
            8. 缓存命中结果到 Redis

        参数:
            query (str): 用户查询文本。
            threshold (float): Softmax 分数阈值，默认 0.85。
                              分数越高表示匹配越精确。

        返回:
            tuple[str|None, bool]:
                - (answer, False): 找到高置信度匹配，answer 为答案文本
                - (None, True):   未找到匹配或置信度不足，需要 RAG 继续处理
                - (None, False):  查询参数无效

        注意:
            返回的 False/True 第二元素表示 "是否需要进入 RAG 流程"：
            - False: 已找到可靠答案，不需要 RAG
            - True:  需要 RAG 进行语义检索
        """
        # ---- 验证输入有效性 ----
        # 查询必须是非空字符串
        if not query or not isinstance(query, str):
            self.logger.error('无效查询: 查询必须为非空字符串')
            return None, True  # 无效查询降级到 RAG 处理

        # ---- 检查 BM25 索引是否就绪 ----
        if self.bm25 is None:
            self.logger.error('BM25模型未初始化，无数据可搜索')
            return None, True  # 无答案，需要 RAG

        # ---- 检查 Redis 查询缓存 ----
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            self.logger.info(f'从Redis缓存获取答案:{query}')
            return cached_answer, False  # 缓存命中，直接返回

        try:
            # ---- 步骤1: 中文分词 ----
            query_tokens = preprocess_text(query)

            # ---- 步骤2: BM25 计算分数 ----
            # get_scores 返回每个文档与查询的 BM25 分数
            scores = self.bm25.get_scores(query_tokens)

            # ---- 步骤3: Softmax 归一化 ----
            softmax_scores = self._softmax(scores)

            # ---- 步骤4: 获取最佳匹配 ----
            best_idx = softmax_scores.argmax()  # 分数最高的文档索引
            best_score = softmax_scores[best_idx]  # 最高分

            # ---- 步骤5: 阈值判断 ----
            if best_score >= threshold:
                # 从统一为字符串列表的 original_questions 中获取问题文本
                original_question = self.original_questions[best_idx]
                # 在 MySQL 中查找对应答案
                answer = self.mysql_client.fetch_answer(original_question)

                if answer:
                    # ---- 缓存命中结果 ----
                    # 将找到的答案缓存到 Redis，加速后续相同查询
                    self.redis_client.set_data(f'answer:{query}', answer)
                    self.logger.info(
                        f'从MySQL获取答案成功:{original_question},'
                        f'softmax相似度:{best_score:.4f}'
                    )
                    return answer, False  # 找到答案，不需要 RAG

            # ---- 步骤6: 置信度不足 ----
            self.logger.warning(
                f'无可靠答案,最高softmax相似度:{best_score:.4f}'
            )
            return None, True  # 无可靠答案，需要 RAG

        except Exception as e:
            # 捕获所有异常，确保系统不会因检索错误而崩溃
            self.logger.error(f'搜索失败:{e}')
            return None, True  # 出错时降级到 RAG 处理
