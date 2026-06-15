"""
图片 OCR 加载器模块
====================
最简单的文档加载器，直接对图片文件执行 OCR 文字识别。

适用场景:
    - 扫描版文档的照片/截图
    - 课件中的图表截图
    - 黑板板书的照片
    - 任何包含文字的图片文件

支持的图片格式:
    - .jpg / .jpeg (JPEG)
    - .png  (PNG)
    - 其他 RapidOCR 支持的格式

使用限制:
    - 仅适用于包含可识别文字的图片
    - 图片质量直接影响 OCR 准确率
    - 复杂排版（多栏、表格）的识别效果有限
"""

from typing import Iterator                         # 类型注解
from edu_ocr import get_ocr                         # OCR 引擎
from langchain_core.documents import Document         # LangChain Document
from langchain_core.document_loaders import BaseLoader  # 基类


class OCRIMGLoader(BaseLoader):
    """
    图片 OCR 加载器，直接对图片文件执行文字识别。

    遵循 LangChain BaseLoader 接口。

    属性:
        img_path (str): 图片文件路径。

    使用示例:
        >>> loader = OCRIMGLoader(img_path="/path/to/photo.png")
        >>> docs = loader.load()
        >>> print(docs[0].page_content)  # OCR 识别结果
    """

    def __init__(self, img_path: str) -> None:
        """
        初始化图片 OCR 加载器。

        参数:
            img_path (str): 图片文件的路径。
        """
        self.img_path = img_path

    def lazy_load(self) -> Iterator[Document]:
        """
        延迟加载模式：对图片执行 OCR 后返回单个 Document。

        Returns:
            Iterator[Document]
        """
        line = self.img2text()
        yield Document(page_content=line, metadata={"source": self.img_path})

    def img2text(self):
        """
        对图片执行 OCR，返回识别的文字。

        处理流程:
            1. 获取 OCR 引擎实例
            2. 直接传入图片路径给 OCR 引擎
            3. 提取识别结果中的文字行
            4. 以换行符连接所有识别结果

        返回:
            str: OCR 识别的文字，各文字行以换行分隔。
                 若未识别到文字则返回空字符串。

        OCR 返回格式:
            result = [(bbox, text, confidence), ...]
            - bbox: 文字边界框坐标
            - text: 识别的文字内容
            - confidence: 置信度分数
        """
        resp = ""
        ocr = get_ocr()  # 获取 OCR 引擎

        # ocr() 可直接接收图片路径（内部自动读取）
        result, _ = ocr(self.img_path)

        if result:
            # 提取每行识别的文字

            ocr_result = [line[1] for line in result]
            # 以换行连接所有识别行
            resp += "\n".join(ocr_result)

        return resp


if __name__ == '__main__':
    # 测试图片 OCR 加载器
    # 请将路径修改为你本地的图片文件路径
    import sys
    test_file = sys.argv[1] if len(sys.argv) > 1 else './samples/ocr_04.png'
    img_loader = OCRIMGLoader(img_path=test_file)
    doc = img_loader.load()
    print(doc)
