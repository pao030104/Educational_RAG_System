"""
MySQL 数据库客户端模块
=======================
负责管理 MySQL 数据库连接，提供数据表创建、数据导入、查询等操作。
主要操作 jpkb（精品课标）表，存储学科-问题-答案三元组。

表结构 (jpkb):
    - id:           自增主键 (INT AUTO_INCREMENT)
    - subject_name: 学科名称，如 "ai", "java", "bigdata" (VARCHAR(20))
    - question:     问题文本 (VARCHAR(1000))
    - answer:       答案文本 (VARCHAR(1000))

依赖:
    - pymysql: MySQL 数据库驱动
    - pandas:  用于读取 CSV 数据文件
"""

import sys
import os

# ---- 设置项目根目录到 sys.path，确保 base 模块可导入 ----
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.append(root_path)

import pymysql        # MySQL 数据库驱动，提供 PEP 249 兼容的数据库接口
import pandas as pd   # 数据分析库，用于读取和解析 CSV 文件
from base import Config, logger  # 导入全局配置和日志器


class MySQLClient:
    """
    MySQL 数据库操作客户端

    负责与 MySQL 数据库的所有交互，包括:
        - 建立和关闭数据库连接
        - 创建数据表 (jpkb)
        - 从 CSV 文件批量导入数据
        - 查询问题和答案

    属性:
        logger:         日志记录器
        connection:     pymysql 数据库连接对象
        cursor:         数据库游标，用于执行 SQL 语句

    使用示例:
        >>> client = MySQLClient()
        >>> client.create_table()
        >>> client.insert_data("data.csv")
        >>> questions = client.fetch_questions()
        >>> answer = client.fetch_answer("什么是AI?")
        >>> client.close()
    """

    def __init__(self):
        """
        建立与 MySQL 数据库的连接。

        使用 Config 中读取的数据库参数（主机、用户名、密码、数据库名）。
        连接成功后会创建一个游标对象，用于后续的 SQL 执行。

        异常:
            pymysql.MySQLError: 当数据库连接失败时（主机不可达、认证失败等），
                               记录错误日志并重新抛出异常。
        """
        self.logger = logger  # 绑定日志器

        try:
            # 使用 Config 中的参数建立 MySQL 连接
            # connect() 返回一个 Connection 对象，代表与数据库的会话
            self.connection = pymysql.connect(
                host=Config().MYSQL_HOST,        # 数据库服务器地址
                user=Config().MYSQL_USER,        # 登录用户名
                password=Config().MYSQL_PASSWORD, # 登录密码
                database=Config().MYSQL_DATABASE  # 默认使用的数据库
            )
            # 创建一个游标对象，用于执行 SQL 语句和获取结果
            # cursor 是默认的元组游标，fetchone/fetchall 返回元组
            self.cursor = self.connection.cursor()
            self.logger.info('连接MySQL数据库成功')
        except pymysql.MySQLError as e:
            self.logger.error('连接MySQL数据库失败:%s' % e)
            raise  # 重新抛出异常，让调用者决定如何处理

    def create_table(self):
        """
        创建 jpkb（精品课标）数据表。

        表结构:
            - id (INT, 自增主键): 记录的唯一标识
            - subject_name (VARCHAR(20)): 学科名称
            - question (VARCHAR(1000)): 问题文本
            - answer (VARCHAR(1000)): 答案文本

        使用 CREATE TABLE IF NOT EXISTS，即使重复执行也不会出错。
        """
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS jpkb(
            id INT AUTO_INCREMENT PRIMARY KEY,
            subject_name VARCHAR(20),
            question VARCHAR(1000),
            answer VARCHAR(1000))
        '''
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()  # DDL 语句也需要显式提交
            self.logger.info('创建表成功')
        except pymysql.MySQLError as e:
            self.logger.error('创建表失败:%s' % e)
            raise

    def insert_data(self, csv_path):
        """
        从 CSV 文件批量导入数据到 jpkb 表。

        CSV 文件格式要求:
            必需列: 学科名称, 问题, 答案
            编码: 需与 pandas read_csv 默认兼容（通常为 UTF-8）

        处理流程:
            1. 使用 pandas 读取 CSV 文件
            2. 逐行遍历数据，提取"学科名称"、"问题"、"答案"三列
            3. 执行 INSERT 语句将每行数据写入数据库
            4. 每条插入后立即提交（小批量场景），出错时回滚

        参数:
            csv_path (str): CSV 数据文件的路径。

        异常:
            Exception: 当文件不存在、格式不匹配或数据库写入失败时抛出。
        """
        try:
            # 使用 pandas 读取 CSV 文件到 DataFrame
            data = pd.read_csv(csv_path)

            # 逐行遍历 DataFrame，iterrows() 返回 (行索引, 行数据) 元组
            for _, row in data.iterrows():
                # 参数化 SQL 查询，使用 %s 占位符防止 SQL 注入
                insert_query = "INSERT INTO jpkb(subject_name, question, answer) VALUES (%s, %s, %s)"
                self.cursor.execute(
                    insert_query,
                    (row['学科名称'], row['问题'], row['答案'])
                )
                self.connection.commit()  # 逐条提交（适合小数据量）
                self.logger.info('插入数据成功')
        except Exception as e:
            self.logger.error('插入数据失败:%s' % e)
            self.connection.rollback()  # 出错时回滚所有未提交的修改
            raise

    def fetch_questions(self):
        """
        获取 jpkb 表中的所有问题。

        此方法用于 BM25 索引的初始化阶段，一次性加载所有问题文本。

        返回:
            list[tuple]: 问题列表，每个元素是单元素元组 (问题文本,)。
                         查询失败时返回空列表。
        """
        try:
            self.cursor.execute('SELECT question FROM jpkb')
            results = self.cursor.fetchall()
            self.logger.info('获取所有问题成功')
            return results
        except pymysql.MySQLError as e:
            self.logger.error('查询失败:%s' % e)
            return []

    def fetch_answer(self, question):
        """
        根据问题文本查询对应的答案。

        用于 BM25 匹配成功后获取精确答案。

        参数:
            question (str): 问题文本，需与数据库中存储的完全一致。

        返回:
            str | None: 对应的答案文本，若问题不存在则返回 None。

        注意:
            参数使用单元素元组 (question,) 传递给 execute，确保
            pymysql 将 question 作为单个参数而非字符序列处理。
        """
        try:
            # 注意: (question,) 是单元素元组，(question) 只是带括号的字符串
            self.cursor.execute(
                'SELECT answer FROM jpkb WHERE question=%s',
                (question,)
            )
            result = self.cursor.fetchone()  # 获取第一条匹配行
            # fetchone 返回元组如 ("答案文本",)，取第一个元素
            # 若未匹配到任何行，result 为 None
            return result[0] if result else None
        except pymysql.MySQLError as e:
            self.logger.error('答案获取失败:%s' % e)
            return None

    def close(self):
        """
        关闭数据库连接，释放资源。

        应在程序退出前或不再需要数据库操作时调用，避免连接泄漏。
        推荐使用 try...finally 或在上下文管理器中调用。
        """
        try:
            self.connection.close()  # 关闭连接，释放 MySQL 服务器端的资源
            self.logger.info('关闭数据库连接成功')
        except pymysql.MySQLError as e:
            self.logger.error('关闭数据库连接失败:%s' % e)


# ==================== 模块直接运行入口 ====================
if __name__ == '__main__':
    # 当直接执行本文件时，演示完整的建表 → 导入流程
    mysql_client = MySQLClient()
    mysql_client.create_table()
    # 构造 CSV 文件路径：相对于当前文件的 ../data/JP学科知识问答.csv
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'JP学科知识问答.csv'
    )
    mysql_client.insert_data(csv_path)
