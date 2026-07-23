"""
RAG (Retrieval-Augmented Generation) 系统核心模块
==================================================
整合查询分类、策略选择、混合检索和 LLM 答案生成的完整 RAG 流程。

RAG 全流程:

    ┌──────────────────────────────────────────────────────────────────┐
    │ 用户查询                                                           │
    │   ↓                                                               │
    │ [1] BERT 查询分类 ──→ 通用知识? ──→ LLM 直接回答                    │
    │   ↓ 专业咨询                                                       │
    │ [2] LLM 策略选择 ──→ 直接检索 / HyDE / 子查询 / 回溯                │
    │   ↓                                                               │
    │ [3] 执行检索策略 ──→ Milvus 混合检索 + BGE-Reranker 精排            │
    │   ↓                                                               │
    │ [4] LLM 上下文生成 ──→ 检索到的文档 + 原始问题 → 最终答案            │
    └──────────────────────────────────────────────────────────────────┘

四种检索策略详解:
    - 直接检索:    原始查询 → 混合检索 → 结果
    - HyDE:       原始查询 → LLM假答案 → 用假答案检索 → 结果
    - 子查询:     原始查询 → LLM拆分为N个子查询 → 分别检索 → 合并去重 → 结果
    - 回溯问题:    原始查询 → LLM简化为基础问题 → 用基础问题检索 → 结果
"""

import sys
import os

# ---- 设置 Python 模块搜索路径 ----
current_dir = os.path.dirname(os.path.abspath(__file__))       # core/
rag_qa_path = os.path.dirname(current_dir)                      # rag_qa/
project_root = os.path.dirname(rag_qa_path)                     # 项目根目录
sys.path.insert(0, rag_qa_path)
sys.path.insert(0, project_root)

from .prompts import RAGPrompts                       # Prompt 模板集合
import time                                          # 性能计时
from base import logger, Config                       # 日志和配置
from .query_classifier import QueryClassifier         # BERT 查询分类器
from .strategy_selector import StrategySelector       # LLM 策略选择器

conf = Config()                                       # 全局配置实例


class RAGSystem:
    """
    RAG 检索增强生成系统

    整合了 RAG 管线的全部环节，是 rag_qa 子系统的顶层控制类。

    属性:
        vector_store (VectorStore):          Milvus 向量存储与检索引擎
        llm (callable):                      LLM 调用函数（非流式），供内部策略使用
        rag_prompt (PromptTemplate):         RAG 答案生成的 Prompt 模板
        model_path (str):                    BERT 查询分类器模型路径
        query_classifier (QueryClassifier):  查询意图分类器（BERT）
        strategy_selector (StrategySelector): 检索策略选择器（LLM）

    使用示例:
        >>> from rag_qa import VectorStore, RAGSystem
        >>> vs = VectorStore()
        >>> rag = RAGSystem(vs, my_llm_function)
        >>> answer = rag.generate_answer("AI学科学费多少？")
    """

    def __init__(self, vector_store, llm):
        """
        初始化 RAG 系统。

        参数:
            vector_store (VectorStore): 已初始化的向量存储实例。
            llm (callable): LLM 调用函数，签名应为 fn(prompt: str) -> str。
                           用于内部策略（HyDE/子查询/回溯）和最终答案生成。
        """
        self.vector_store = vector_store
        self.llm = llm
        # 加载 RAG 答案生成的 Prompt 模板
        self.rag_prompt = RAGPrompts.rag_prompt()

        # 初始化 BERT 查询分类器（加载微调后的模型）
        self.model_path = os.path.join(current_dir, "bert_query_classifier")
        self.query_classifier = QueryClassifier(model_path=self.model_path)

        # 初始化 LLM 策略选择器
        self.strategy_selector = StrategySelector()

    # ==================== 三种增强检索策略 ====================

    def _retrieve_with_hyde(self, query):
        """
        HyDE (Hypothetical Document Embeddings) 策略。

        原理:
            用户查询通常很短（例如"AI的应用"），而知识库文档很长。
            短查询和长文档在语义空间中距离较远，直接检索效果差。
            HyDE 先用 LLM 生成一段"假设的答案"，再用这段详细文本去检索，
            弥合查询-文档的语义鸿沟。

        流程:
            用户查询 → LLM 生成假设答案 → 用假设答案检索 → 返回文档列表

        参数:
            query (str): 用户原始查询。

        返回:
            list[Document]: 检索到的文档列表，失败时返回空列表。
        """
        logger.info(f"使用HyDE策略进行检索(查询:{query})")

        # 获取 HyDE 专用 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt()

        try:
            # 调用 LLM 生成假设答案
            hypo_answer = self.llm(
                hyde_prompt_template.format(query=query)
            ).strip()
            logger.info(f"HyDE生成的假设答案:{hypo_answer}")

            # 用假设答案而非原始查询进行检索
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer, k=conf.RETRIEVAL_K
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []

    def _retrieve_with_subqueries(self, query):
        """
        子查询检索策略。

        原理:
            复杂查询包含多个子问题，每个子问题可能匹配不同的文档。
            例如"比较 Java 和 Python" → 拆为"Java 的特点"和"Python 的特点"，
            分别检索两个子查询，合并结果。

        流程:
            用户查询 → LLM 拆分为 N 个子查询 → 每个子查询独立检索
            → 合并所有结果 → 按内容去重 → 返回合并后的文档列表

        参数:
            query (str): 用户原始查询。

        返回:
            list[Document]: 去重合并后的文档列表，失败时返回空列表。
        """
        logger.info(f"使用子查询进行检索(查询：{query})")

        # 获取子查询分解 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt()

        try:
            # 调用 LLM 生成子查询列表
            subqueries_text = self.llm(
                subquery_prompt_template.format(query=query)
            ).strip()

            # 按换行符拆分为子查询列表，过滤空行
            subqueries = [
                q.strip() for q in subqueries_text.split("\n") if q.strip()
            ]
            logger.info(f"子查询：{subqueries}")

            if not subqueries:
                logger.warning("没有找到子查询")
                return []

            # 对每个子查询执行检索，合并结果
            all_docs = []
            for sub_q in subqueries:
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=conf.RETRIEVAL_K
                )
                all_docs.extend(docs)
                logger.info(f"子查询 {sub_q} 检索到: {len(docs)}个文档")

            # ---- 去重：基于文档内容 ----
            # 多个子查询可能检索到相同的文档
            unique_docs_dict = {
                doc.page_content: doc for doc in all_docs
            }
            unique_docs = list(unique_docs_dict.values())

            logger.info(
                f"所有子查询共检索到: {len(all_docs)} 个文档,"
                f"去重后还剩: {len(unique_docs)} 个文档"
            )
            return unique_docs

        except Exception as e:
            logger.error(f"检索子查询时出错: {e}")
            return []

    def _retrieve_with_backtracking(self, query):
        """
        回溯问题检索策略。

        原理:
            用户的问题可能包含过多细节或特定约束，导致检索范围过窄。
            通过将复杂问题"回溯"为更基础的版本，扩大检索范围。
            例如: "我有100亿条数据想存到Milvus可以吗？"
                  → 回溯为 "Milvus支持的最大数据规模是多少？"

        流程:
            用户查询 → LLM 生成简化问题 → 用简化问题检索 → 返回文档列表

        参数:
            query (str): 用户原始查询。

        返回:
            list[Document]: 检索到的文档列表，失败时返回空列表。
        """
        logger.info(f"使用回溯问题策略进行检索(查询:{query})")

        # 获取回溯问题 Prompt 模板
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()

        try:
            # 调用 LLM 生成简化版本的问题
            simplified_query = self.llm(
                backtrack_prompt_template.format(query=query)
            ).strip()
            logger.info(f"生成的回溯问题:{simplified_query}")

            # 用简化问题而非原始查询进行检索
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=conf.RETRIEVAL_K
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    # ==================== 检索与合并 ====================

    def retrieve_and_merge(self, query, source_filter=None, strategy=None):
        """
        根据指定策略执行检索，返回最终候选文档。

        此方法是策略到检索的统一入口，根据策略名称分派到对应的检索方法。

        分派逻辑:
            - "回溯问题检索"  → _retrieve_with_backtracking
            - "子查询检索"    → _retrieve_with_subqueries
            - "假设问题检索"  → _retrieve_with_hyde (HyDE)
            - 其他             → 直接检索

        参数:
            query (str): 用户查询文本。
            source_filter (str, optional): 学科类别过滤。
            strategy (str, optional): 策略名称。为 None 时自动选择。

        返回:
            list[Document]: 最终候选文档列表，最多 CANDIDATE_M 个。
        """
        # 若未指定策略，使用策略选择器自动选择
        if strategy is None:
            strategy = self.strategy_selector.select_strategy(query)

        # ---- 策略分派 ----
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._retrieve_with_backtracking(query)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query)
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query)
        else:
            # 默认: 直接检索（包含 "直接检索" 及其他未知策略名）
            logger.info(f"使用直接检索策略(查询: {query})")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter
            )
            # 注意：hybrid_search_with_rerank 返回的是 BGE-Reranker 精排后的父文档

        logger.info(f'策略 "{strategy}" 检索到 {len(ranked_sub_chunks)} 个候选文档')

        # ---- 截取 Top-M 个文档作为最终上下文 ----
        final_context_docs = ranked_sub_chunks[:conf.CANDIDATE_M]
        logger.info(f"最终检索到 {len(final_context_docs)} 个文档作为上下文")

        return final_context_docs

    # ==================== 答案生成 ====================

    def generate_answer(self, query, source_filter=None):
        """
        执行完整的 RAG 流程，生成最终答案。

        完整流程:
            ┌─────────────────────────────────────────────────┐
            │ 1. BERT 查询分类 → 通用知识 / 专业咨询            │
            ├─────────────────────────────────────────────────┤
            │ 2. 若为通用知识: LLM 直接回答（不检索）           │
            ├─────────────────────────────────────────────────┤
            │ 3. 若为专业咨询:                                  │
            │    a. LLM 策略选择                                │
            │    b. 执行对应检索策略                             │
            │    c. 构建上下文                                  │
            │    d. LLM 基于上下文生成答案                       │
            └─────────────────────────────────────────────────┘

        参数:
            query (str): 用户查询文本。
            source_filter (str, optional): 学科类别过滤，如 'ai', 'java'。
                                           为 None 时不过滤。

        返回:
            tuple: (answer_text, sources_list, category)
            - answer_text (str): LLM 生成的答案文本
            - sources_list (list): 文章来源列表
            - category (str): 查询分类（通用知识/专业咨询）

        使用示例:
            >>> answer = rag.generate_answer("AI学科有哪些课程？")
            >>> print(answer)
        """
        start_time = time.time()
        logger.info(f"处理查询:{query},学科过滤:{source_filter}")

        # ---- 步骤1: 查询意图分类 ----
        # BERT 模型判断是"通用知识"（闲聊/常识）还是"专业咨询"（需要检索知识库）
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果:{query_category}(查询:{query})")

        # ---- 步骤2a: 通用知识 → LLM 直接回答 ----
        if query_category == "通用知识":
            logger.info(f"查询为通用知识,使用LLM回答")
            # 构造 Prompt：无上下文，基于 LLM 自身知识回答
            prompt_input = self.rag_prompt.format(
                context="", question=query, phone=conf.CUSTOMER_SERVICE_PHONE
            )
            try:
                answer = self.llm(prompt_input)
            except Exception as e:
                logger.error(f"LLM回答失败:{e}")
                answer = (
                    f"抱歉，处理您的通用知识问题时出错。"
                    f"请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
                )

            processing_time = time.time() - start_time
            logger.info(
                f"通用知识查询处理完成 "
                f"(耗时: {processing_time:.2f}s, 查询: '{query}')"
            )
            return answer, [], query_category

        # ---- 步骤2b: 专业咨询 → 完整 RAG 流程 ----
        logger.info("查询为专业知识查询,执行RAG流程")

        # ---- 步骤3a: LLM 选择检索策略 ----
        strategy = self.strategy_selector.select_strategy(query)

        # ---- 步骤3b: 执行检索 ----
        context_docs = self.retrieve_and_merge(
            query, source_filter=source_filter, strategy=strategy
        )

        # ---- 步骤3c: 提取文章来源 ----
        sources = list(set(
            doc.metadata.get('source', '未知来源') for doc in context_docs if hasattr(doc, 'metadata')
        )) if context_docs else []
        if not sources:
            sources = list(set(
                str(doc.metadata.get('file_path', doc.metadata.get('source', '未知来源')))
                for doc in context_docs if hasattr(doc, 'metadata')
            )) if context_docs else []

        # ---- 步骤3d: 构建上下文 ----
        if context_docs:
            # 将多个文档的内容用双换行拼接
            context = "\n\n".join([doc.page_content for doc in context_docs])
            logger.info(f"检索到 {len(context_docs)} 个文档作为上下文, 来源: {sources}")
        else:
            context = ""
            logger.info(f"没有检索到文档作为上下文,上下文为空")

        # ---- 步骤3d: LLM 基于上下文生成答案 ----
        # 填充 RAG Prompt 模板：上下文 + 用户问题 + 客服电话
        prompt_input = self.rag_prompt.format(
            context=context,
            question=query,
            phone=conf.CUSTOMER_SERVICE_PHONE
        )

        try:
            answer = self.llm(prompt_input)
        except Exception as e:
            logger.error(f"RAG流程执行失败:{e}")
            answer = (
                f"抱歉，处理您的专业知识问题时出错。"
                f"请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
            )

        # ---- 记录处理完成日志 ----
        processing_time = time.time() - start_time
        logger.info(
            f"专业知识查询处理完成 "
            f"(耗时: {processing_time:.2f}s, 查询: '{query}')"
        )
        return answer
