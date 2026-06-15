"""
base 包初始化模块
=================
负责设置 Python 模块搜索路径 (sys.path)，确保项目内的模块可以相互导入。
同时统一导出项目的核心配置类和日志器，供其他模块便捷使用。

导出符号:
    - Config: 全局配置管理类，读取 config.ini 中的所有配置项
    - logger: 全局日志器实例，支持文件和控制台双输出

路径设置说明:
    将项目根目录和 base 目录添加到 sys.path，确保在任何工作目录下
    运行项目时，各模块都能通过 "from base import ..." 正确导入。
"""

import sys
import os

# ---- 将项目根目录添加到 Python 模块搜索路径 ----
# os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 的作用:
#   1. os.path.abspath(__file__)  → 获取本文件 (base/__init__.py) 的绝对路径
#   2. os.path.dirname(...)       → 获取 base 目录的路径 (第一次调用)
#   3. os.path.dirname(...)       → 获取项目根目录的路径 (第二次调用)
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)  # 将根目录加入搜索路径，使 "from base import ..." 等导入可用

# ---- 将 base 目录自身也添加到搜索路径 ----
base_path = os.path.dirname(os.path.abspath(__file__))
if base_path not in sys.path:
    sys.path.append(base_path)  # 确保 base 包内的模块可以相互引用

# ---- 导出项目的核心组件 ----
# 通过这些导入语句，其他模块只需 "from base import Config, logger" 即可获取配置和日志实例
from config import Config      # noqa: E402 (忽略 import 位置警告，因为必须先设置 sys.path)
from logger import logger      # noqa: E402
