"""
PDF 文档加载器模块
===================
基于 PyMuPDF (fitz) 的 PDF 文档加载器，支持文本提取和嵌入式图片 OCR。

工作流程:
    ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │ 打开 PDF 文件  │───→│ 逐页提取文本      │───→│ 提取嵌入式图片     │
    │              │    │ (原生文字层)     │    │ (xref 引用)       │
    └──────────────┘    └──────────────────┘    └──────────────────┘
                                                       │
                                                       ▼
    ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │ 合并输出      │←───│ OCR 识别图片文字   │←───│ 图片大小过滤      │
    │              │    │ (RapidOCR)       │    │ (跳过小图标)      │
    └──────────────┘    └──────────────────┘    └──────────────────┘

图片过滤策略:
    仅对尺寸超过页面一定比例（默认 60%）的图片进行 OCR，避免对
    Logo、小图标等无关图片执行 OCR，节省处理时间。

注意:
    依赖 PyMuPDF 包，安装命令: pip install PyMuPDF
    不要与 fitz 包混淆：import fitz 实际上是从 PyMuPDF 导入的
"""

import cv2                                        # OpenCV，用于图像旋转
import fitz                                       # PyMuPDF，注意不是 pip install fitz
import numpy as np                                # 数值计算，图片数组处理
from PIL import Image                             # 图片对象转换
from tqdm import tqdm                              # 进度条
from typing import Iterator                        # 类型注解
from edu_ocr import get_ocr                        # OCR 引擎工厂函数
from langchain_core.documents import Document       # LangChain Document 类
from langchain_core.document_loaders import BaseLoader  # LangChain 加载器基类

# ---- PDF OCR 过滤阈值 ----
# (图片宽度/页面宽度, 图片高度/页面高度) 的最小比例
# 只有宽高同时超过此比例的图片才会被 OCR 处理
# 目的: 过滤掉 Logo、小图标、装饰图等非内容性的小图片
PDF_OCR_THRESHOLD = (0.6, 0.6)


class OCRPDFLoader(BaseLoader):
    """
    PDF 文档加载器，提取文本并对大图执行 OCR。

    遵循 LangChain BaseLoader 接口，支持 lazy_load (延迟加载) 和 load (立即加载)。

    属性:
        file_path (str): PDF 文件的路径。

    使用示例:
        >>> loader = OCRPDFLoader(file_path="/path/to/document.pdf")
        >>> docs = loader.load()
        >>> for doc in docs:
        ...     print(doc.page_content[:100])  # 打印前100字符
    """

    def __init__(self, file_path: str) -> None:
        """
        初始化 PDF 加载器。

        参数:
            file_path (str): PDF 文件的绝对或相对路径。
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """
        延迟加载模式：逐文档产出，适合大文件场景。

        Returns:
            Iterator[Document]: 一次产出单个 Document 的生成器。
        """
        # 提取 PDF 的全部文本和 OCR 内容
        line = self.pdf2text()
        # 构造 LangChain Document，源文件路径作为元数据
        yield Document(page_content=line, metadata={"source": self.file_path})

    def pdf2text(self):
        """
        从 PDF 提取全部文本内容，包括原生文字和图片 OCR 结果。

        处理流程（逐页）:
            1. 获取页面原生文本层文字
            2. 获取页面中的所有图片元信息
            3. 筛选大尺寸图片（过滤小图标/Logo）
            4. 对符合条件的图片执行 OCR
            5. 将 OCR 文本追加到总输出中

        返回:
            str: PDF 的完整文本内容（原生文字 + OCR 文字）。

        技术细节:
            - page.get_text("text"): 提取文本层内容
            - page.get_image_info(xrefs=True): 获取图片的 xref (交叉引用编号)
            - fitz.Pixmap: 将 PDF 中的图片数据渲染为像素图
            - 页面旋转: 处理 PDF 中的旋转页面，OCR 前先旋转回正常方向
        """
        ocr = get_ocr()  # 获取 OCR 引擎实例

        # 打开 PDF 文件
        doc = fitz.open(self.file_path)
        resp = ""  # 累积全部文本内容

        # 创建进度条，跟踪页面处理进度
        b_unit = tqdm(
            total=doc.page_count,
            desc="OCRPDFLoader context page index: 0"
        )

        # ---- 逐页处理 ----
        for i, page in enumerate(doc):
            # 更新进度条描述
            b_unit.set_description(
                "OCRPDFLoader context page index: {}".format(i)
            )
            b_unit.refresh()

            # ---- 步骤1: 提取原生文本层 ----
            # "text" 模式: 按阅读顺序提取文本
            text = page.get_text("text")
            resp += text + "\n"

            # ---- 步骤2: 获取嵌入图片信息 ----
            # xrefs=True: 同时返回图片的交叉引用编号
            img_list = page.get_image_info(xrefs=True)

            # ---- 步骤3: 逐张处理图片 ----
            for img in img_list:
                # xref (Cross-Reference): PDF 内部对象编号，用于精确定位图片数据
                if xref := img.get("xref"):
                    # 图片在页面上的边界框 [x0, y0, x1, y1]
                    bbox = img["bbox"]

                    # ---- 过滤小图片 ----
                    # 检查图片尺寸是否超过设定的阈值比例
                    if (
                        (bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0]
                        or (bbox[3] - bbox[1]) / (page.rect.height) < PDF_OCR_THRESHOLD[1]
                    ):
                        continue  # 跳过小图片（Logo、图标等）

                    # 从 xref 渲染图片像素数据
                    pix = fitz.Pixmap(doc, xref)

                    # ---- 处理页面旋转 ----
                    # PDF 页面可能被旋转存储，OCR 前需要旋转回正常方向
                    if int(page.rotation) != 0:
                        # 将像素数据转为 numpy 数组
                        img_array = np.frombuffer(
                            pix.samples, dtype=np.uint8
                        ).reshape(pix.height, pix.width, -1)

                        # numpy → PIL → OpenCV 格式转换
                        tmp_img = Image.fromarray(img_array)
                        ori_img = cv2.cvtColor(
                            np.array(tmp_img), cv2.COLOR_RGB2BGR
                        )

                        # 反向旋转恢复原始方向 (360 - 旋转角度)
                        rot_img = self.rotate_img(
                            img=ori_img, angle=360 - page.rotation
                        )
                        # OpenCV BGR → RGB 再转回 numpy 数组
                        img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                    else:
                        img_array = np.frombuffer(
                            pix.samples, dtype=np.uint8
                        ).reshape(pix.height, pix.width, -1)

                    # ---- 步骤4: OCR 识别图片文字 ----
                    # result: [(bbox, text, confidence), ...]
                    # _: 时间统计数据（模型优化用）
                    result, _ = ocr(img_array)

                    if result:
                        # 提取识别出的文字（忽略位置和置信度）
                        ocr_result = [line[1] for line in result]
                        resp += "\n".join(ocr_result)

            # 更新进度条
            b_unit.update(1)

        return resp

    def rotate_img(self, img, angle):
        """
        旋转图片以校正 PDF 中的旋转页面。

        旋转原理:
            - 使用 OpenCV 的仿射变换 (Affine Transform)
            - 绕图片中心旋转指定角度
            - 自动扩展画布以容纳旋转后的完整图片

        参数:
            img (np.ndarray): OpenCV 格式的原始图片数组 (H, W, C)。
            angle (float):    旋转角度（度），正值逆时针，负值顺时针。

        返回:
            np.ndarray: 旋转后的图片数组，画布已扩展。
        """
        h, w = img.shape[:2]  # 获取图片高和宽

        # 旋转中心点：图片正中心
        rotate_center = (w / 2, h / 2)

        # 获取2D旋转矩阵
        # 参数: (旋转中心, 角度, 缩放因子)
        # 角度正值=逆时针，负值=顺时针
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)

        # ---- 计算旋转后的新边界 ----
        # 旋转后图片尺寸会变大（对角线变为新的宽或高）
        new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
        new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))

        # 调整旋转矩阵的平移分量，使旋转后的图片居中
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        # 执行仿射变换旋转
        rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
        return rotated_img


# ==================== 模块自测 ====================
if __name__ == '__main__':
    # 测试 PDF 加载器
    # 请将路径修改为你本地的 PDF 文件路径
    import sys
    test_file = sys.argv[1] if len(sys.argv) > 1 else './samples/ocr_03.pdf'
    pdf_loader = OCRPDFLoader(file_path=test_file)
    doc = pdf_loader.load()
    print(type(doc))
    print(doc)
