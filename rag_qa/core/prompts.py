"""
Prompt 模板管理模块
====================
统一管理 RAG 系统中所有 LLM 提示词 (Prompt) 模板。

Prompt 工程是 RAG 系统的关键环节，良好的提示词可以显著提升：
    - 答案的准确性和相关性
    - 检索策略选择的合理性
    - 查询分解和简化的质量

本模块包含以下 Prompt 模板:
    1. RAG 答案生成模板 (rag_prompt):
       指导 LLM 基于检索到的上下文生成答案，无法回答时引导用户联系客服。

    2. HyDE 假设答案模板 (hyde_prompt):
       生成假设答案用于检索，解决查询与文档语义不匹配的问题。

    3. 子查询分解模板 (subquery_prompt):
       将复杂查询拆分为多个简单子查询，分别检索再合并结果。

    4. 回溯问题模板 (backtracking_prompt):
       将复杂查询简化为更基础的问题，提高检索命中率。
"""

from langchain.prompts import PromptTemplate  # LangChain 的提示词模板类


class RAGPrompts:
    """
    RAG 提示词模板集合

    所有方法均为静态方法，无需实例化即可使用。
    每个模板方法返回一个 PromptTemplate 对象，通过 .format(**kwargs) 填充变量。

    使用示例:
        >>> template = RAGPrompts.rag_prompt()
        >>> prompt = template.format(context="...", question="...", phone="123")
    """

    @staticmethod
    def rag_prompt():
        """
        RAG 答案生成提示词模板。

        指导原则:
            - 优先基于检索到的上下文回答问题
            - 若无上下文则基于 LLM 自身知识回答
            - 无法回答时提供客服电话作为降级方案
            - 若答案来源于检索文档，需在回答中说明（提高可信度）

        变量:
            context (str): 检索到的文档上下文，空字符串表示无上下文
            question (str): 用户的原始问题
            phone (str): 客服电话号码

        返回:
            PromptTemplate: 包含 template 和 input_variables 的模板对象。
        """
        return PromptTemplate(
            template="""
            你是一个智能助手，帮助用户回答问题。
            如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
            如果答案来源于检索到的文档，请在回答中说明。

            上下文: {context}
            问题: {question}

            如果无法回答，请回复："信息不足，无法回答，请联系人工客服，电话：{phone}。"
            回答:
            """,
            input_variables=["context", "question", "phone"],
        )

    @staticmethod
    def hyde_prompt():
        """
        HyDE (Hypothetical Document Embeddings) 假设答案生成模板。

        HyDE 策略的核心思想:
            用户查询往往过于简短，与知识库中详细文档的语义空间不匹配。
            通过让 LLM 先生成一个"假设的答案"，再用这个假设答案去检索，
            可以弥合查询-文档之间的语义鸿沟，提高检索召回率。

        适用于:
            查询较为抽象、描述性强的场景。
            例如: "人工智能在教育领域有哪些应用？" → 生成一段描述 → 用描述去检索

        变量:
            query (str): 用户的原始查询

        返回:
            PromptTemplate
        """
        return PromptTemplate(
            template="""
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：
            问题: {query}
            假设答案:
            """,
            input_variables=["query"],
        )

    @staticmethod
    def subquery_prompt():
        """
        子查询分解模板。

        策略原理:
            复杂查询可能包含多个子问题，一次性检索效果不佳。
            将复杂查询拆分为多个简单子查询，每个子查询独立检索，
            最后合并去重，可以覆盖问题的多个方面。

        适用于:
            查询涉及多个实体或需要多方面信息比较的场景。
            例如: "比较 Java 和 Python 的优缺点" → 拆为 "Java 的优点"、"Java 的缺点"、
                  "Python 的优点"、"Python 的缺点" 四个子查询

        变量:
            query (str): 用户的复杂查询

        返回:
            PromptTemplate
        """
        return PromptTemplate(
            template="""
            将以下复杂查询分解为多个简单子查询，每行一个子查询：
            查询: {query}
            子查询:
            """,
            input_variables=["query"],
        )

    @staticmethod
    def backtracking_prompt():
        """
        回溯问题简化模板。

        策略原理:
            用户的问题可能包含过多细节或特定约束条件，导致检索范围过窄。
            通过将复杂问题"回溯"为更基础、更通用的版本，扩大检索范围，
            然后再结合原始需求筛选结果。

        适用于:
            查询带有许多具体约束条件，直接检索结果太少或没有的场景。
            例如: "我有一个包含 100 亿条记录的 Milvus 集群..." →
                  回溯为 "Milvus 支持的最大数据规模是多少？"

        变量:
            query (str): 用户的复杂查询

        返回:
            PromptTemplate
        """
        return PromptTemplate(
            template="""
            将以下复杂查询简化为一个更简单的问题：
            查询: {query}
            简化问题:
            """,
            input_variables=["query"],
        )
