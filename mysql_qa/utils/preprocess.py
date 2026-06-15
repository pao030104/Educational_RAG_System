"""
文本预处理工具模块
===================
提供中文文本预处理功能，主要用于分词。
使用 jieba 分词库，支持精确模式和全模式。

分词在 BM25 检索中的角色:
    BM25 算法基于词袋模型（Bag of Words），需要将原始文本分解为词元 (token)。
    对于中文而言，词语之间没有空格分隔，因此需要借助 jieba 等分词工具
    将连续的中文文本切分为有意义的词语序列。
"""

import jieba          # 结巴中文分词库，提供精确模式分词
from base import logger  # 全局日志器


def preprocess_text(text):
    """
    对输入文本进行预处理：转小写 → jieba 分词。

    处理步骤:
        1. text.lower(): 将英文字母统一转为小写，避免大小写差异影响匹配
        2. jieba.lcut(): 使用 jieba 的精确模式分词，返回词语列表

    参数:
        text (str): 待预处理的原始文本。

    返回:
        list[str]: 分词后的词语列表。若输入不是字符串（如 None），则返回空列表。

    异常处理:
        当 text 不是 str 类型时（例如 None 或数字），调用 .lower() 会引发
        AttributeError。此时记录错误日志并返回空列表，确保程序不会崩溃。

    使用示例:
        >>> result = preprocess_text("人工智能是什么？")
        >>> print(result)
        ['人工智能', '是', '什么', '？']
    """
    logger.info('开始预处理文本')
    try:
        # 先统一转小写（对英文有效，中文无影响），再进行分词
        # jieba.lcut 返回列表，如 "人工智能" → ['人工智能']
        return jieba.lcut(text.lower())
    except AttributeError as e:
        # 当 text 不是字符串时，.lower() 方法不存在
        # 例如 text 为 None 或 int 类型
        logger.error(f'预处理文本失败:{e}')
        return []
