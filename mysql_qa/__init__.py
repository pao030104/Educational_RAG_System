"""
mysql_qa 包 - MySQL 关键词匹配问答子系统
=========================================
基于 BM25 算法的关键词匹配问答系统，适用于已有明确问答对的知识库场景。
结合 Redis 缓存加速和 MySQL 持久化存储，提供快速的精确匹配能力。

模块组成:
    - db.mysql_client:   MySQL 数据库客户端，管理数据表的 CRUD 操作
    - cache.redis_client: Redis 缓存客户端，缓存分词结果和查询答案
    - retrieval.bm25_search: BM25 关键词检索引擎，计算查询与知识库的相似度
    - utils.preprocess:  文本预处理工具，支持中文分词等操作

典型用法:
    from mysql_qa import MySQLClient, RedisClient, BM25Search

    mysql_cli = MySQLClient()                    # 连接 MySQL
    redis_cli = RedisClient()                    # 连接 Redis
    bm25 = BM25Search(redis_cli, mysql_cli)      # 初始化检索引擎
    answer, need_rag = bm25.search("什么是AI?")   # 执行检索
"""

# ---- 导出 mysql_qa 子系统的核心组件 ----
# 采用相对导入，从子模块中导入公开类，统一对外暴露
from .db.mysql_client import MySQLClient      # MySQL 数据库操作客户端
from .cache.redis_client import RedisClient   # Redis 缓存操作客户端
from .retrieval.bm25_search import BM25Search # BM25 关键词匹配检索引擎
