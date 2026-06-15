"""
日志系统模块
============
提供统一的日志记录功能，支持同时输出到控制台和文件。
使用 Python 标准库 logging 模块，配置为 INFO 级别，按时间戳格式化输出。
"""

# 导入 Python 标准日志库，提供灵活的日志记录功能
import logging
# 导入路径操作库，用于创建日志文件所在目录
import os
# 导入配置类，获取日志文件路径等配置信息
from base.config import Config


def setup_logging(log_file=Config().LOG_FILE):
    """
    初始化日志系统，配置日志器 (Logger)、处理器 (Handler) 和格式化器 (Formatter)。

    设计要点:
        1. 自动创建日志文件所在的目录（如果不存在）
        2. 使用 logger.handlers 检查避免重复添加处理器（防止多次初始化产生重复日志）
        3. 同时配置两个处理器：
           - FileHandler: 将日志写入文件，持久化存储，便于事后排查
           - StreamHandler: 将日志输出到控制台，便于实时监控
        4. 统一使用 INFO 级别，过滤掉 DEBUG 调试信息
        5. 日志格式为: "时间 - 日志器名 - 级别 - 消息内容"

    参数:
        log_file (str): 日志文件的完整路径，默认从 Config 中读取。

    返回:
        logging.Logger: 配置完成的日志器实例，名称为 'EduRAG'。

    使用示例:
        >>> logger = setup_logging()
        >>> logger.info("系统启动成功")
        >>> logger.error("发生错误: 数据库连接失败")
    """
    # 确保日志文件所在的目录存在
    # os.path.dirname 提取日志文件的目录部分，os.makedirs 递归创建目录
    # exist_ok=True 表示目录已存在时不会抛出异常
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 获取名为 'EduRAG' 的日志器实例
    # getLogger 是单例模式，多次调用返回同一个 logger 实例
    logger = logging.getLogger('EduRAG')

    # 设置日志器的全局最低级别为 INFO
    # 低于 INFO 的 DEBUG 级别日志将被完全忽略
    logger.setLevel(logging.INFO)

    # ---- 防止重复添加处理器 ----
    # 如果 logger 已经有处理器（之前初始化过），则跳过添加步骤
    # 这避免了在模块重载或多线程环境下出现重复日志条目
    if not logger.handlers:
        # ---- 文件处理器：将日志持久化写入文件 ----
        # FileHandler 负责将日志记录写入指定的文件
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)  # 文件处理器也设为 INFO 级别

        # ---- 控制台处理器：将日志实时输出到终端 ----
        # StreamHandler 默认输出到 sys.stderr，便于开发者实时查看
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # 控制台处理器也设为 INFO 级别

        # ---- 日志格式化器：定义日志条目的输出模板 ----
        # %(asctime)s: 日志产生的时间戳（年-月-日 时:分:秒,毫秒）
        # %(name)s: 日志器名称（即 'EduRAG'）
        # %(levelname)s: 日志级别（INFO / WARNING / ERROR）
        # %(message)s: 日志消息正文
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 将格式化器绑定到处理器
        file_handler.setFormatter(formatter)       # 文件日志使用此格式
        console_handler.setFormatter(formatter)     # 控制台日志使用相同格式

        # 将处理器添加到日志器
        # 一个 logger 可以有多个 handler，每条日志会同时传递给所有 handler
        logger.addHandler(file_handler)    # 添加文件输出
        logger.addHandler(console_handler)  # 添加控制台输出

    # 返回配置完成的日志器，供其他模块调用
    return logger


# ---- 模块级别初始化 ----
# 在模块导入时自动执行 setup_logging，创建全局可用的 logger 实例
# 其他模块通过 "from base import logger" 即可直接使用
logger = setup_logging()
