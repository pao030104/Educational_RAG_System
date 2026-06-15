"""
阿里达摩院文档语义分割器模块
==============================
基于阿里达摩院开源的 nlp_bert_document-segmentation_chinese-base 模型
进行文档级别的语义分割。该模型能够识别文档中的主题边界，实现比规则分割
更准确的语义级文档切分。

模型介绍:
    论文: https://arxiv.org/abs/2107.09278
    任务: 文档分割 (Document Segmentation) — 将长文档按语义主题切分为多个段落
    模型: BERT-based，在中文文档语料上微调
    用途: 识别文档中"话题转换"的位置，在语义边界处切分

适用场景:
    - 长文档的自动章节划分
    - PDF/PPT 转换文本后的语义重构
    - 需要对文档内容进行语义理解后再切分的场景

与 ChineseRecursiveTextSplitter 的区别:
    - ChineseRecursiveTextSplitter: 基于规则（标点符号）的递归分割，速度快但缺乏语义理解
    - AliTextSplitter: 基于深度学习模型的语义分割，更准确但速度较慢

性能提示:
    模型在 CPU 上运行，处理速度较慢。对于大多数场景，
    推荐使用 ChineseRecursiveTextSplitter 作为默认切分器。
    仅在需要精确语义分割时使用本模块。
"""

from langchain.text_splitter import CharacterTextSplitter  # 基类文本分割器
import re                                                  # 正则表达式
from typing import List                                    # 类型注解
from modelscope.pipelines import pipeline                  # ModelScope 模型管线


class AliTextSplitter(CharacterTextSplitter):
    """
    基于阿里达摩院语义分割模型的中文文本分割器

    继承自 CharacterTextSplitter，使用深度学习模型替换基于规则的切分逻辑。

    属性:
        pdf (bool): 是否为 PDF 源文本。PDF 文本通常包含更多格式噪音，
                    需额外预处理（去除多余换行、空白等）。

    模型位置:
        rag_qa/nlp_bert_document-segmentation_chinese-base/
        （相对于本模块文件自动计算路径）

    使用示例:
        >>> splitter = AliTextSplitter(pdf=False)
        >>> segments = splitter.split_text("这是一段很长的文档内容...")
        >>> for seg in segments:
        ...     print(seg)
    """

    def __init__(self, pdf: bool = False, **kwargs):
        """
        初始化语义分割器。

        参数:
            pdf (bool): 文本来源是否为 PDF。PDF 文本需要额外的格式清理。
                        默认 False。
            **kwargs: 传递给父类 CharacterTextSplitter 的额外参数。
        """
        super().__init__(**kwargs)
        self.pdf = pdf  # 标记是否为 PDF 源文本

        # ---- 加载语义分割模型（仅一次） ----
        # 使用 ModelScope 线上的模型 ID，首次运行会自动下载模型（约 400MB），
        # 缓存于 ~/.cache/modelscope/hub/ 目录下。
        # 模型: iic/nlp_bert_document-segmentation_chinese-base
        #   - 阿里达摩院开源的文档语义分割模型
        #   - 基于 BERT 架构，在中文文档语料上微调
        # device="cpu": 在 CPU 上运行（可改为 "cuda:0" 使用 GPU）
        self._pipeline = pipeline(
            task="document-segmentation",
            model="iic/nlp_bert_document-segmentation_chinese-base",
            device="cpu"
        )

    def split_text(self, text: str) -> List[str]:
        """
        使用 BERT 语义模型对文本进行分割。

        处理流程:
            1. 若为 PDF 文本: 清理多余换行、压缩空白
            2. 使用已加载的达摩院 BERT 文档分割模型
            3. 运行模型推理，获取语义边界
            4. 按模型输出的边界切分文本

        参数:
            text (str): 待分割的原始文本。

        返回:
            list[str]: 按语义边界切分后的文本段列表。

        注意事项:
            - 模型通过 ModelScope pipeline 加载，首次调用需下载模型
            - 模型在 CPU 上运行，输入过长时速度较慢
            - 需要安装: pip install "modelscope[nlp]"
        """
        # ---- PDF 文本预处理 ----
        # PDF 转换的文本通常包含大量格式噪音
        if self.pdf:
            # 将3个及以上连续换行压缩为1个换行
            text = re.sub(r"\n{3,}", r"\n", text)
            # 将所有空白字符（空格、制表符等）压缩为单个空格
            text = re.sub('\s', " ", text)
            # 去除双换行（PDF 中的段落标记）
            text = re.sub("\n\n", "", text)

        # ---- 执行语义分割 ----
        # 模型输出格式: {"text": "分段1\\n\\t分段2\\n\\t分段3"}
        # 分段间用 \\n\\t 分隔
        result = self._pipeline(documents=text)

        # 按 \\n\\t 分割模型输出，过滤空字符串
        sent_list = [i for i in result["text"].split("\n\t") if i]

        return sent_list


# ==================== 模块自测 ====================
if __name__ == '__main__':
    # 创建语义分割器实例（非 PDF 模式）
    model_split = AliTextSplitter()

    # 测试文本：移动端语音唤醒模型的描述文档
    test_text = (
        '移动端语音唤醒模型，检测关键词为"小云小云"。'
        '模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。'
        '模型输入为Fbank特征，输出为基于char建模的中文全集token预测，'
        '测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。'
        '模型训练采用"basetrain + finetune"的模式，'
        'basetrain过程使用大量内部移动端数据，'
        '在此基础上，使用1万条设备端录制安静场景"小云小云"数据进行微调，'
        '得到最终面向业务的模型。'
        '后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，'
        '得到新的语音唤醒模型，但暂时未开放模型finetune功能。'
    )

    result = model_split.split_text(text=test_text)
    print(result)
