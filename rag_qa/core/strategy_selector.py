"""
检索策略选择器模块
===================
使用 LLM 分析用户查询的特征，自动选择最适合的检索策略。

四种检索策略:

    ┌──────────────────────────────────────────────────────────────┐
    │ 策略名称         │ 描述                         │ 适用场景      │
    ├──────────────────────────────────────────────────────────────┤
    │ 直接检索         │ 直接使用原始查询进行检索       │ 查询意图明确   │
    │ 假设问题检索     │ 生成假设答案后基于答案检索     │ 查询较抽象     │
    │ (HyDE)          │ 弥合查询-文档语义鸿沟          │               │
    │ 子查询检索       │ 拆分为多个子查询分别检索       │ 查询涉及多方面  │
    │ 回溯问题检索     │ 简化为更基础的问题后检索       │ 查询过于具体   │
    └──────────────────────────────────────────────────────────────┘

选择流程:
    用户查询 → LLM 分析 → 返回策略名称 → RAGSystem 执行对应策略
"""

import sys
import os

# ---- 设置 Python 模块搜索路径 ----
current_dir = os.path.dirname(os.path.abspath(__file__))       # core/
rag_qa_path = os.path.dirname(current_dir)                      # rag_qa/
project_root = os.path.dirname(rag_qa_path)                     # 项目根目录
sys.path.insert(0, rag_qa_path)
sys.path.insert(0, project_root)

from langchain.prompts import PromptTemplate   # 提示词模板
from base import logger, Config                 # 日志和配置
from openai import OpenAI                       # OpenAI 兼容 API 客户端


class StrategySelector:
    """
    检索策略智能选择器

    使用 LLM 分析查询语义，从四种策略中选出最优方案。
    通过精心设计的 Few-Shot Prompt 引导 LLM 做出准确判断。

    属性:
        client (OpenAI):               LLM API 客户端
        strategy_prompt_tmplate (PromptTemplate): 策略选择提示词模板

    使用示例:
        >>> selector = StrategySelector()
        >>> strategy = selector.select_strategy("AI学科学费是多少？")
        >>> print(strategy)  # 输出: "直接检索"
    """

    def __init__(self):
        """
        初始化策略选择器，建立 LLM 连接并加载 Prompt 模板。
        """
        # 初始化 OpenAI 兼容客户端（连接 DashScope API）
        self.client = OpenAI(
            api_key=Config().DASHSCOPE_API_KEY,
            base_url=Config().DASHSCOPE_BASE_URL,
        )
        # 加载策略选择的 Prompt 模板（含 Few-Shot 示例）
        self.strategy_prompt_tmplate = self._get_strategy_prompt()

    def call_dashscope(self, prompt):
        """
        调用 DashScope LLM API，获取策略选择结果。

        使用低温度 (temperature=0.1) 以获得更确定性的输出，
        提高策略选择的一致性和可靠性。

        参数:
            prompt (str): 填充了查询的策略选择提示词。

        返回:
            str | None: LLM 返回的策略名称；API 调用失败时返回 None。
        """
        try:
            completion = self.client.chat.completions.create(
                model=Config().LLM_MODEL,
                messages=[
                    {"role": "system", "content": '你是一个有用的助手'},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # 低温度 = 更确定性的输出，适合分类/选择任务
                timeout=30,
            )
            # 安全提取响应内容
            if (completion.choices
                    and completion.choices[0].message
                    and completion.choices[0].message.content):
                return completion.choices[0].message.content
            return None
        except Exception as e:
            logger.error(f"DashScope API 调用失败: {e}")
            # 降级策略：失败时默认使用最通用的直接检索
            return '直接检索'

    def _get_strategy_prompt(self):
        """
        构建策略选择的 Few-Shot Prompt 模板。

        Few-Shot 设计原则:
            1. 每种策略包含清晰的描述和适用场景说明
            2. 每类策略提供 1-2 个具体示例
            3. 示例覆盖不同学科和查询类型
            4. 明确要求只返回策略名称，不输出冗长的分析过程

        返回:
            PromptTemplate: 策略选择的提示词模板对象。
        """
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    * 示例：
                        * 查询:AI 学科学费是多少？
                        * 策略：直接检索
                    * 查询:JAVA的课程大纲是什么?
                        * 策略：直接检索
            2.  **假设问题检索(HyDE):**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：人工智能在教育领域的应用有哪些？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：比较 Milvus 和 Zilliz Cloud 的优缺点。
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """,
            input_variables=["query"],
        )

    def select_strategy(self, query):
        """
        为给定的用户查询选择最优检索策略。

        处理流程:
            1. 将查询填入策略选择 Prompt 模板
            2. 调用 LLM 获取策略建议
            3. 安全处理返回值（strip 空白，None 降级为默认策略）

        参数:
            query (str): 用户查询文本。

        返回:
            str: 策略名称，可能的值:
                 - "直接检索"
                 - "假设问题检索"
                 - "子查询检索"
                 - "回溯问题检索"
                 默认为 "直接检索"

        使用示例:
            >>> ss = StrategySelector()
            >>> strategy = ss.select_strategy("周杰伦在世界的影响力有多大")
            >>> print(strategy)
        """
        # 构造完整的策略选择 Prompt
        result = self.call_dashscope(
            self.strategy_prompt_tmplate.format(query=query)
        )
        # 安全处理：strip 去除首尾空白和换行，None 时降级为直接检索
        strategy = result.strip() if result else '直接检索'

        logger.info(f'为查询{query}选择的策略是：{strategy}')
        return strategy


if __name__ == "__main__":
    # 测试策略选择器
    ss = StrategySelector()
    ss.select_strategy("周杰伦在世界的影响力有多大")
