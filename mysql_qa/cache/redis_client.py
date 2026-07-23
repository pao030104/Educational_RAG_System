"""
Redis 缓存客户端模块
=====================
负责管理 Redis 数据库连接，提供数据缓存和读取功能。
主要用于缓存预计算的中文分词结果和热门查询的答案，减少重复计算和 MySQL 访问压力。

缓存策略:
    - qa_original_questions:  原始问题文本列表（从 MySQL 加载后缓存）
    - qa_tokenized_questions: 预分词结果（避免每次启动重新分词）
    - answer:{query}:         查询答案缓存（加速重复查询）

缓存数据流:
    首次启动 → MySQL 加载数据 → 分词处理 → 存入 Redis
    后续启动 → Redis 直接读取（跳过 MySQL 和分词步骤）
"""

import sys
import os

# ---- 设置项目根目录到 sys.path，确保 base 模块可导入 ----
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.append(root_path)

import redis           # Redis Python 客户端
import json            # JSON 序列化/反序列化，用于缓存复杂数据结构
import socket          # 用于快速检测 Redis 端口是否可用
from base import Config, logger  # 全局配置和日志


class RedisClient:
    """
    Redis 缓存操作客户端

    封装了 Redis 的常用操作，支持 JSON 序列化的数据存取。
    使用 StrictRedis 客户端，启用 decode_responses 自动将字节转为字符串。

    属性:
        logger: 日志记录器
        client: redis.StrictRedis 客户端实例

    缓存数据格式:
        所有通过 set_data 存储的值都会经过 json.dumps 序列化，
        通过 get_data 读取的值会经过 json.loads 反序列化。
        这确保了列表、字典等复杂类型可以正确存取。

    使用示例:
        >>> redis_cli = RedisClient()
        >>> redis_cli.set_data("key1", [1, 2, 3])
        >>> data = redis_cli.get_data("key1")  # 返回 [1, 2, 3]
        >>> answer = redis_cli.get_answer("什么是AI?")
    """

    def __init__(self):
        """
        建立与 Redis 服务器的连接。

        使用 Config 中的参数：主机、端口、密码、数据库编号。
        启用 decode_responses=True，使 get/set 操作自动进行
        bytes ↔ str 转换，简化代码。

        异常:
            redis.RedisError: 连接失败时记录错误并重新抛出。
        """
        self.logger = logger  # 绑定日志器

        try:
            conf = Config()
            try:
                with socket.create_connection((conf.REDIS_HOST, conf.REDIS_PORT), timeout=0.5):
                    pass
            except OSError as e:
                self.client = None
                self.logger.warning(f"Redis 未启动，缓存功能已禁用: {e}")
                return

            # StrictRedis 是 Redis 的标准客户端类
            # decode_responses=True: 自动将 Redis 返回的 bytes 解码为 str
            self.client = redis.StrictRedis(
                host=conf.REDIS_HOST,            # Redis 服务器地址
                port=conf.REDIS_PORT,            # Redis 端口号
                password=conf.REDIS_PASSWORD,    # Redis 认证密码
                db=conf.REDIS_DB,                # 使用的数据库编号（默认 0）
                socket_connect_timeout=1,        # Redis 未启动时快速降级
                socket_timeout=1,                # 缓存读写最多等待 1 秒
                protocol=2,                       # 兼容 Windows 旧版 Redis，不发送 HELLO 3
                decode_responses=True            # 自动字符串解码
            )
            self.logger.info('连接Redis数据库成功')
        except redis.RedisError as e:
            self.logger.error('连接Redis数据库失败:%s' % e)
            raise  # 缓存非核心依赖？此处选择抛出，让上层决定是否降级

    def set_data(self, key, value):
        """
        将数据以 JSON 格式存入 Redis。

        自动对 value 进行 json.dumps 序列化，支持字典、列表等复杂类型。

        参数:
            key (str): 缓存键名。
            value (any): 要缓存的值，必须可被 json.dumps 序列化。

        注意:
            若存储失败，仅记录错误日志而不会抛出异常，
            因为缓存失败不应影响主业务流程。
        """
        if self.client is None:
            return
        try:
            # json.dumps 将 Python 对象转换为 JSON 字符串
            self.client.set(key, json.dumps(value))
            self.logger.info(f'储存数据到Redis:{key}成功')
        except redis.RedisError as e:
            self.logger.error(f'储存数据到Redis:{key}失败:{e}')

    def get_data(self, key):
        """
        从 Redis 获取 JSON 格式存储的数据并反序列化。

        参数:
            key (str): 缓存键名。

        返回:
            any | None: 反序列化后的 Python 对象；若键不存在或读取失败返回 None。

        注意:
            与 set_data 配对使用：set_data 做 json.dumps → get_data 做 json.loads。
        """
        if self.client is None:
            return None
        try:
            data = self.client.get(key)  # 从 Redis 获取字符串
            # 只有 data 不为 None 时才进行 JSON 反序列化
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f'从Redis获取数据失败:%s' % e)
            return None

    def get_answer(self, query):
        """
        根据查询文本获取缓存的答案。

        此方法专用于答案缓存场景。与 set_data 配合使用：
            - 存储: self.set_data(f'answer:{query}', answer)
            - 读取: self.get_answer(query)

        参数:
            query (str): 用户查询文本。

        返回:
            str | None: 缓存的答案文本；无缓存或读取失败返回 None。

        注意:
            返回前会通过 json.loads 解码，确保与 set_data 的 json.dumps 编码
            保持一致，避免返回带 JSON 引号的字符串。
        """
        if self.client is None:
            return None
        try:
            # 答案缓存使用统一的前缀 "answer:" 命名空间
            answer = self.client.get(f'answer:{query}')
            if answer:
                self.logger.info(f'从Redis获取答案:{query}')
                # json.loads 解码：因为存储时使用了 json.dumps
                return json.loads(answer)
            return None
        except (redis.RedisError, json.JSONDecodeError) as e:
            # 同时捕获 Redis 异常和 JSON 解析异常
            self.logger.error(f'从Redis查询失败:{e}')
            return None


if __name__ == '__main__':
    # 测试 Redis 连接
    redcli = RedisClient()
    print(redcli)
