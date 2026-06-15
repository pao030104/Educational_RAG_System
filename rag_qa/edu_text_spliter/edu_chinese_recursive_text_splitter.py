"""
中文递归文本分割器模块
========================
基于 LangChain RecursiveCharacterTextSplitter 的中文优化版本。

核心改进:
    1. 中文标点优先分割: 先按换行、再按句号/感叹号/问号、再按分号、最后按逗号递归切分
    2. 从文本末尾分割: 与 LangChain 默认的从头分割不同，从末尾开始分割更符合中文阅读习惯
    3. 可选保留分隔符: keep_separator=True 时保留分隔符在切分结果中，避免丢失标点语义

分割优先级（从高到低）:
    1. "\\n\\n"    - 段落分隔（双换行）
    2. "\\n"       - 行分隔（单换行）
    3. "。|！|？"  - 句子结尾标点
    4. "\\.\\s|\\!\\s|\\?\\s" - 英文标点 + 空格
    5. "；|;\\s"   - 分号（子句分隔）
    6. "，|,\\s"   - 逗号（短语分隔）

算法流程:
    对每个待切分的文本块:
        1. 按优先级最高的分隔符尝试切分
        2. 切分后的小块如果仍然超过 chunk_size，用下一级分隔符递归切分
        3. 合并能在 chunk_size 内的小块
        4. 对于单个超长且无法切分的块（如超长英文单词），保留原样
"""

import re                                                    # 正则表达式，用于文本分隔
from typing import List, Optional, Any                        # 类型注解
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 基类
import logging                                               # 日志

logger = logging.getLogger(__name__)                          # 模块级日志器


def _split_text_with_regex_from_end(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    """
    使用正则表达式从文本末尾开始分割，支持保留分隔符。

    与 str.split 的区别:
        - 支持正则表达式分隔符
        - 从末尾开始分割（更适合中文阅读习惯）
        - 可选保留分隔符在结果中（keep_separator=True）

    参数:
        text (str):           待分割的文本。
        separator (str):      正则表达式分隔符模式。
        keep_separator (bool): True 时保留分隔符在切分结果中，False 时丢弃。

    返回:
        list[str]: 分割后的文本片段列表，已过滤空字符串。

    示例:
        >>> _split_text_with_regex_from_end("句1。句2。句3", "。", True)
        ['句1。', '句2。', '句3']
    """
    if separator:
        if keep_separator:
            # 使用捕获组 () 在 re.split 结果中保留分隔符
            # 产生的列表格式为: [文本1, 分隔符1, 文本2, 分隔符2, ...]
            _splits = re.split(f"({separator})", text)

            # 将文本和紧随其后的分隔符合并
            # zip(_splits[0::2], _splits[1::2]) 将文本与分隔符配对
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]

            # 如果分裂后元素数为奇数，意味着最后一段没有分隔符，直接添加
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            # 不保留分隔符的简单分割
            splits = re.split(separator, text)
    else:
        # 无分隔符时退化为按字符分割
        splits = list(text)

    # 过滤掉空字符串
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """
    中文递归文本分割器

    继承自 LangChain 的 RecursiveCharacterTextSplitter，专为中文文本优化。

    核心特性:
        - 中文标点优先: 按句号/感叹号/问号等中文标点切分，保持语义完整性
        - 递归分割: 大块用高级别标点切，不够再用低级别标点，逐级递归
        - 从末尾开始: 与中文的"从后往前读"习惯一致
        - 后处理: 去除多余换行和首尾空白

    属性（继承自父类）:
        chunk_size:  每个块的目标大小（字符数）
        chunk_overlap: 相邻块重叠的字符数
        keep_separator: 是否在结果中保留分隔符
        is_separator_regex: 分隔符是否为正则表达式

    使用示例:
        >>> splitter = ChineseRecursiveTextSplitter(
        ...     chunk_size=300, chunk_overlap=50
        ... )
        >>> chunks = splitter.split_text("这是一个很长的中文字符串。包含了很多信息...")
        >>> for chunk in chunks:
        ...     print(chunk)
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        初始化中文递归文本分割器。

        参数:
            separators (list[str], optional): 自定义分隔符优先级列表。
                                              默认按中文标点优先级排列。
            keep_separator (bool): True 保留分隔符，False 丢弃。默认 True。
            is_separator_regex (bool): True 分隔符为正则表达式。默认 True。
            **kwargs: 传递给父类 RecursiveCharacterTextSplitter 的额外参数，
                      包括 chunk_size, chunk_overlap 等。
        """
        # 调用父类构造器，传递 keep_separator 和其他参数
        super().__init__(keep_separator=keep_separator, **kwargs)

        # ---- 设置分隔符优先级列表 ----
        # 从高到低排列，越靠前的分隔符优先级越高
        # 每个分隔符都是正则表达式模式
        self._separators = separators or [
            "\n\n",           # 第1优先级: 段落间空行
            "\n",             # 第2优先级: 换行
            "。|！|？",       # 第3优先级: 中文句尾标点
            "\.\s|\!\s|\?\s", # 第4优先级: 英文句尾 + 空格
            "；|;\s",          # 第5优先级: 分号（子句分隔）
            "，|,\s"           # 第6优先级: 逗号（短语分隔）
        ]

        self._is_separator_regex = is_separator_regex  # 标记分隔符是否为正则

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        递归分割文本的核心方法（重写父类方法）。

        递归逻辑:
            1. 在当前分隔符列表中，找到第一个能在 text 中匹配到的分隔符
            2. 用该分隔符切分文本
            3. 对切分后小于 chunk_size 的片段进行合并
            4. 对大于 chunk_size 的片段，用剩余分隔符列表递归调用此方法

        参数:
            text (str): 待分割的文本。
            separators (list[str]): 当前可用的分隔符列表。

        返回:
            list[str]: 分割完成后的文本块列表。
        """
        final_chunks = []  # 最终的分割结果

        # ---- 步骤1: 找到最合适的当前级分隔符 ----
        # 从 separators 列表的最后一个（最低优先级）开始，向前搜索
        # 找到第一个在文本中匹配到的分隔符
        separator = separators[-1]  # 初始为最低优先级
        new_separators = []         # 剩余未使用的分隔符（供递归使用）

        for i, _s in enumerate(separators):
            # 为正则模式添加转义保护
            _separator = _s if self._is_separator_regex else re.escape(_s)

            if _s == "":
                # 空字符串分隔符 = 按字符分割，直接使用
                separator = _s
                break

            if re.search(_separator, text):
                # 在文本中找到了当前分隔符，使用它
                separator = _s
                # 剩余的分隔符列表传给下一层递归使用
                new_separators = separators[i + 1:]
                break

        # ---- 步骤2: 用选中分隔符执行分割 ----
        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(
            text, _separator, self._keep_separator
        )

        # ---- 步骤3: 合并小于 chunk_size 的片段，递归处理大于的片段 ----
        _good_splits = []  # 当前累积的待合并小片段
        # 合并时使用的连接符：保留分隔符时用空串（分隔符已包含），否则用分隔符本身
        _separator = "" if self._keep_separator else separator

        for s in splits:
            if self._length_function(s) < self._chunk_size:
                # 片段小于 chunk_size，加入待合并队列
                _good_splits.append(s)
            else:
                # 片段大于 chunk_size
                if _good_splits:
                    # 先合并并输出之前累积的小片段
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []

                if not new_separators:
                    # 没有更多分隔符了，即使片段仍超大也只能原样保留
                    final_chunks.append(s)
                else:
                    # 递归调用：用下一级分隔符继续切分这个超大片段
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)

        # ---- 步骤4: 处理最后累积的小片段 ----
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)

        # ---- 步骤5: 后处理 ----
        # 去除多余换行（超过2个连续换行 → 压缩为1个）、去除首尾空白、过滤空串
        return [
            re.sub(r"\n{2,}", "\n", chunk.strip())
            for chunk in final_chunks
            if chunk.strip() != ""
        ]


# ==================== 模块自测 ====================
if __name__ == "__main__":
    # 创建一个中文分割器实例
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,         # 保留分隔符
        is_separator_regex=True,     # 使用正则分隔
        chunk_size=150,              # 每个块最大150字符
        chunk_overlap=10             # 相邻块重叠10字符
    )

    # 测试文本：一篇较长中文文章
    ls = [
        """中国对外贸易形势报告（75页）。前 10 个月，一般贸易进出口 19.5 万亿元，增长 25.1%， 比整体进出口增速高出 2.9 个百分点，占进出口总额的 61.7%，较去年同期提升 1.6 个百分点。其中，一般贸易出口 10.6 万亿元，增长 25.3%，占出口总额的 60.9%，提升 1.5 个百分点；进口8.9万亿元，增长24.9%，占进口总额的62.7%， 提升 1.8 个百分点。加工贸易进出口 6.8 万亿元，增长 11.8%， 占进出口总额的 21.5%，减少 2.0 个百分点。""",
    ]

    # 逐篇测试分割效果
    for inum, text in enumerate(ls):
        print(inum)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            print(chunk)
            print('*' * 80)  # 分割线
