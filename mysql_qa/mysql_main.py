"""
MySQL 问答子系统 - 独立运行入口
===============================
提供 mysql_qa 子系统的独立命令行测试界面，不依赖 RAG 模块。
可用于验证 MySQL 数据库连接、Redis 缓存和 BM25 检索功能是否正常。

运行方式:
    python -m mysql_qa.mysql_main
    或
    cd mysql_qa && python mysql_main.py

功能:
    1. 初始化 MySQL 连接、Redis 缓存和 BM25 检索引擎
    2. 进入交互式 REPL 循环
    3. 仅使用关键词匹配（不调用 RAG），适用于已有明确问答对的场景
"""

# ---- 导入 mysql_qa 子系统的核心组件 ----
from db.mysql_client import MySQLClient        # MySQL 数据库客户端
from retrieval.bm25_search import BM25Search   # BM25 检索引擎
from cache.redis_client import RedisClient     # Redis 缓存客户端
from base import logger                         # 全局日志器
import time                                     # 用于计时


class MySQLQASystem:
    """
    MySQL 关键词问答系统 - 独立版本

    仅使用 BM25 关键词匹配回答问题，不包含 RAG 语义检索。
    适用于已有明确问答对、查询措辞与知识库高度一致的使用场景。

    属性:
        logger:       日志记录器
        mysql_client: MySQL 数据库客户端
        redis_client: Redis 缓存客户端
        bm25_search:  BM25 关键词检索引擎

    使用示例:
        >>> qa = MySQLQASystem()
        >>> answer = qa.query("AI学科课程大纲是什么？")
        >>> print(answer)
    """

    def __init__(self):
        """初始化 MySQL 问答系统的各个组件。"""
        self.logger = logger
        self.mysql_client = MySQLClient()      # 建立 MySQL 连接
        self.redis_client = RedisClient()      # 建立 Redis 连接
        # BM25 检索引擎在构造时自动加载数据并构建索引
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    def query(self, query):
        """
        执行关键词匹配查询（不包含 RAG 语义检索）。

        查询流程:
            1. 记录开始时间
            2. 调用 bm25_search.search 进行关键词匹配
            3. 若匹配成功，返回答案；否则返回兜底提示

        参数:
            query (str): 用户查询文本。

        返回:
            str: 匹配到的答案或 "SQL未找到答案" 兜底消息。
        """
        start_time = time.time()  # 记录查询开始时间
        self.logger.info(f'处理查询:{query}')

        # search 返回 (answer, need_rag)，这里忽略 need_rag
        # 因为本系统不集成 RAG，不论是否需要都只使用 BM25 结果
        answer, _ = self.bm25_search.search(query, threshold=0.85)

        if answer:
            self.logger.info(f'从MySQL获取答案:{query}')
        else:
            self.logger.info(f'从MySQL获取答案失败,需要调用RAG系统')
            answer = 'SQL未找到答案'

        # 记录处理耗时（精确到小数点后两位）
        processing_time = time.time() - start_time
        self.logger.info(f'处理时间:{processing_time:.2f}秒')

        return answer


# ==================== 命令行入口 ====================

def main():
    """
    主函数 - 提供 MySQL 问答子系统的独立命令行界面。

    交互流程:
        1. 初始化 MySQLQASystem
        2. 进入 REPL 循环
        3. 用户输入查询 → 系统返回 BM25 匹配结果
        4. 输入 'exit' 退出
    """
    # 初始化系统
    mysql_system = MySQLQASystem()

    try:
        # 打印欢迎信息
        print(f'\n欢迎使用MySQL系统!')
        print("输入查询进行问答，输入'exit'退出系统")

        # ---- REPL 主循环 ----
        while True:
            query = input('\n请输入查询:').strip()

            # 检查退出条件
            if query.lower() == 'exit':
                logger.info('退出MySQL系统')
                print('\n退出系统')
                break

            # 执行查询并输出结果
            answer = mysql_system.query(query)
            print(f'\n答案:{answer}')

    except KeyboardInterrupt:
        # 用户按下 Ctrl+C
        logger.info('用户通过 Ctrl+C 退出系统')
        print('\n\n退出系统')
    except Exception as e:
        # 捕获所有未预期异常
        logger.error(f'系统错误:{e}')
        print(f'\n发生错误:{e}')
    finally:
        # 确保关闭 MySQL 连接，释放资源
        mysql_system.mysql_client.close()


# Python 标准入口
if __name__ == '__main__':
    main()
