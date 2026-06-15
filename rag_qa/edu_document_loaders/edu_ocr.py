"""
OCR 引擎工厂模块
==================
提供统一的 OCR 识别引擎获取接口，自动选择最优的 OCR 后端。

后端选择策略:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 优先尝试 rapidocr_paddle (PaddlePaddle)                        │
    │    - 优点: 支持 GPU 加速，速度快                                  │
    │    - 安装: pip install rapidocr-paddle                            │
    │    - 要求: 已安装 PaddlePaddle                                    │
    │                                                                   │
    │ 2. 降级到 rapidocr_onnxruntime (ONNX Runtime)                     │
    │    - 优点: CPU 优化，资源占用低，安装简单                          │
    │    - 安装: pip install rapidocr-onnxruntime                       │
    │    - 要求: 已安装 ONNX Runtime                                    │
    └─────────────────────────────────────────────────────────────────┘

RapidOCR 简介:
    基于 PaddleOCR 的轻量级 OCR 引擎，支持:
        - 文字检测 (Text Detection): 定位图片中的文字区域
        - 文字识别 (Text Recognition): 识别区域内的文字内容
        - 文字分类 (Text Classification): 判断文字方向（竖排/横排）
"""

from typing import TYPE_CHECKING  # 类型检查时的条件导入


def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    """
    获取 OCR 引擎实例，自动选择最优后端。

    尝试顺序:
        1. rapidocr_paddle: 支持 det_use_cuda、cls_use_cuda、rec_use_cuda 参数
        2. rapidocr_onnxruntime: 无 GPU 参数，纯 CPU 运行

    参数:
        use_cuda (bool): 是否启用 GPU 加速（仅 rapidocr_paddle 支持）。
                         默认为 True。若 GPU 不可用，Paddle 会自动降级到 CPU。

    返回:
        RapidOCR: RapidOCR 实例，调用方式为 ocr(image_path_or_array)。

    使用示例:
        >>> ocr = get_ocr()
        >>> result, _ = ocr("path/to/image.png")
        >>> for box, text, confidence in result:
        ...     print(f"识别文字: {text}, 置信度: {confidence:.2f}")

    异常:
        本函数不抛出异常：若两种后端都不可用，会由 ImportError 向上传播。
    """
    try:
        # ---- 尝试方案1: PaddlePaddle 后端 ----
        from rapidocr_paddle import RapidOCR
        # 分别控制检测、分类、识别三个阶段是否使用 GPU 加速
        ocr = RapidOCR(
            det_use_cuda=use_cuda,  # 文字检测: GPU 加速
            cls_use_cuda=use_cuda,  # 文字方向分类: GPU 加速
            rec_use_cuda=use_cuda   # 文字识别: GPU 加速
        )
    except ImportError:
        # ---- 降级方案2: ONNX Runtime 后端 ----
        # PaddlePaddle 未安装时使用 ONNX Runtime（安装更轻量）
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()  # ONNX 版本不支持 GPU 参数，默认 CPU

    return ocr
