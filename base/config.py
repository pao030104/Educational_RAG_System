"""
配置管理模块
============
负责读取和管理整个系统的所有配置项，包括数据库连接、LLM API、检索参数等。
配置通过 config.ini 文件进行集中管理，支持默认回退值和环境变量覆盖。

配置优先级（从高到低）:
    1. 环境变量（如 MYSQL_HOST, DASHSCOPE_API_KEY 等）
    2. config.ini 配置文件
    3. 代码中的 fallback 默认值

环境变量映射:
    MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
    REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB
    MILVUS_HOST, MILVUS_PORT, MILVUS_DATABASE_NAME, MILVUS_COLLECTION_NAME
    LLM_MODEL, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL
    PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE, CHUNK_OVERLAP
    RETRIEVAL_K, CANDIDATE_M
    VALID_SOURCES, CUSTOMER_SERVICE_PHONE, LOG_FILE
"""

# 导入 Python 标准配置解析库，用于读取 .ini 格式配置文件
import configparser
# 导入路径操作库，用于处理文件和目录路径
import os
# 导入 ast.literal_eval 替代 eval()，安全解析 Python 字面量（列表、字典等），避免代码注入风险
import ast


class Config:
    """
    全局配置管理类

    负责从 config.ini 文件中读取所有系统配置参数。
    使用 configparser 解析 INI 格式文件，并为每个配置项提供类型安全的属性访问。
    所有配置项均设有 fallback 默认值，确保在配置文件缺失或格式错误时系统仍可运行。

    配置分类:
        - [mysql]: MySQL 数据库连接参数
        - [redis]: Redis 缓存数据库连接参数
        - [milvus]: Milvus 向量数据库连接参数
        - [llm]: 大语言模型 API 配置
        - [retrieval]: 文档检索相关参数
        - [app]: 应用级别配置（学科类别、客服电话等）
        - [logger]: 日志系统配置

    使用示例:
        >>> conf = Config()
        >>> print(conf.MYSQL_HOST)        # 输出: localhost
        >>> print(conf.RETRIEVAL_K)       # 输出: 5
    """

    def __init__(self, config_file='../config.ini'):
        """
        初始化配置管理器，加载并解析配置文件。

        参数:
            config_file (str): 配置文件相对于本模块文件的路径，默认为 '../config.ini'。
                              路径会被解析为绝对路径，因此不受当前工作目录 (CWD) 影响。

        工作原理:
            1. 获取本模块文件 (config.py) 所在的绝对目录
            2. 将相对路径 config_file 与模块目录拼接，得到配置文件的绝对路径
            3. 使用 configparser 读取并解析 INI 文件（UTF-8 编码）
            4. 逐项读取各配置段下的具体参数，赋予类型安全的默认回退值
        """
        # ---- 步骤1: 加载配置文件 ----
        # 创建 ConfigParser 实例，用于解析 INI 格式的配置文件
        self.config = configparser.ConfigParser()
        # 获取本模块文件所在目录的绝对路径（例如: /path/to/project/base/）
        module_dir = os.path.dirname(os.path.abspath(__file__))
        # 将相对路径与模块目录拼接并规范化，得到配置文件的绝对路径
        # 例如: module_dir + '../config.ini' → /path/to/project/config.ini
        config_path = os.path.normpath(os.path.join(module_dir, config_file))
        # 读取配置文件内容，指定 UTF-8 编码以支持中文字符
        self.config.read(config_path, encoding='utf-8')

        # ---- 步骤2: MySQL 数据库配置 ----
        # 以下配置项定义 MySQL 数据库连接所需的主机、用户名、密码和数据库名
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')            # MySQL 主机地址
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')                 # MySQL 用户名
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='')             # MySQL 密码（请在 config.ini 或环境变量中设置）
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='subject_kg')    # MySQL 数据库名称

        # ---- 步骤3: Redis 缓存配置 ----
        # 以下配置项定义 Redis 数据库连接参数，用于缓存查询结果和预计算数据
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')       # Redis 主机地址
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)           # Redis 端口号（整数类型）
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='')        # Redis 访问密码（请在 config.ini 或环境变量中设置）
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)                  # Redis 数据库编号（整数类型，默认 0）

        # ---- 步骤4: Milvus 向量数据库配置 ----
        # 以下配置项定义 Milvus 向量数据库连接和集合参数
        self.MILVUS_URI = self.config.get('milvus', 'uri', fallback='')                         # Milvus Lite 本地文件路径，为空时使用远程服务
        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='localhost')              # Milvus 服务主机地址
        self.MILVUS_PORT = self.config.getint('milvus', 'port', fallback=19530)                 # Milvus 服务端口号
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name', fallback='milvus_qa')   # Milvus 数据库名称
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='milvus_qa') # Milvus 集合（类似表）名称

        # ---- 步骤5: 大语言模型 (LLM) 配置 ----
        # 以下配置项定义调用 LLM API 所需的模型名称和认证信息
        self.LLM_MODEL = self.config.get('llm', 'model_name', fallback='qwen3.7-plus')           # LLM 模型名称（如 qwen3.7-plus）
        self.DASHSCOPE_API_KEY = self.config.get('llm', 'dashscope_api_key', fallback='')        # DashScope API 密钥
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url', fallback='')      # DashScope API 基础 URL

        # ---- 步骤6: 文档检索参数配置 ----
        # 以下配置项控制 RAG 系统中文档切分和检索的行为
        # 采用父子块（Parent-Child Chunk）策略：先用大块切分保留上下文，再小块检索提高精度
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)  # 父块切分大小（字符数）
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)     # 子块切分大小（字符数）
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)            # 相邻块之间的重叠字符数
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)                 # 混合检索返回的候选文档数
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)                 # 最终选入上下文的文档数

        # ---- 步骤7: 应用级别配置 ----
        # 以下配置项定义应用层面的设置
        # 使用 ast.literal_eval 安全地解析 INI 文件中的 Python 列表字面量
        # 例如: "['ai','java','test','ops','bigdata']" → Python 列表对象
        self.VALID_SOURCES = ast.literal_eval(
            self.config.get('app', 'valid_source', fallback='["mysql"]')
        )
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='')  # 客服电话
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='./logs/app.log')                      # 日志文件存储路径

        # ---- 步骤8: 环境变量覆盖（优先级高于 config.ini） ----
        # 如果设置了对应的环境变量，则覆盖 config.ini 中的值。
        # 这样可以在不修改 config.ini 的情况下注入敏感配置（如 API Key、密码），
        # 特别适合 CI/CD 和容器化部署场景。
        self._apply_env_overrides()

    def validate(self):
        """
        验证关键配置项是否已正确设置，在系统启动时尽早发现配置问题。

        检查项:
            - LLM API Key 是否已配置
            - LLM Base URL 是否已配置
            - 块大小参数是否合理（child <= parent, overlap < child）
            - 检索参数是否合理（candidate_m <= retrieval_k）

        返回:
            list[str]: 警告信息列表，若为空表示所有配置正常。
        """
        warnings = []
        # LLM 配置检查
        if not self.DASHSCOPE_API_KEY:
            warnings.append("DASHSCOPE_API_KEY 未设置，LLM 调用将失败")
        if not self.DASHSCOPE_BASE_URL:
            warnings.append("DASHSCOPE_BASE_URL 未设置，LLM 调用将失败")
        # 块大小合理性检查
        if self.CHILD_CHUNK_SIZE > self.PARENT_CHUNK_SIZE:
            warnings.append(
                f"CHILD_CHUNK_SIZE ({self.CHILD_CHUNK_SIZE}) 大于 "
                f"PARENT_CHUNK_SIZE ({self.PARENT_CHUNK_SIZE})，"
                f"子块不应大于父块"
            )
        if self.CHUNK_OVERLAP >= self.CHILD_CHUNK_SIZE:
            warnings.append(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) >= "
                f"CHILD_CHUNK_SIZE ({self.CHILD_CHUNK_SIZE})，"
                f"可能导致无限切分或空块"
            )
        # 检索参数合理性检查
        if self.CANDIDATE_M > self.RETRIEVAL_K:
            warnings.append(
                f"CANDIDATE_M ({self.CANDIDATE_M}) > "
                f"RETRIEVAL_K ({self.RETRIEVAL_K})，"
                f"最终候选数不应大于检索召回数"
            )
        return warnings

    @staticmethod
    def _env_int(name: str, fallback: int) -> int:
        """安全地将字符串环境变量转换为整数，非法值时回退到 fallback。"""
        raw = os.environ.get(name)
        if raw is None:
            return fallback
        try:
            return int(raw)
        except (ValueError, TypeError):
            logger = __import__('logging').getLogger('EduRAG')
            logger.warning(
                f"环境变量 {name}={raw!r} 不是有效的整数，"
                f"使用回退值 {fallback}"
            )
            return fallback

    def _apply_env_overrides(self):
        """
        使用环境变量覆盖配置值。

        环境变量优先级高于 config.ini 中的值，
        主要用于注入敏感信息（密码、API Key），避免在配置文件中明文存储。
        """
        # MySQL 配置
        self.MYSQL_HOST = os.environ.get('MYSQL_HOST', self.MYSQL_HOST)
        self.MYSQL_USER = os.environ.get('MYSQL_USER', self.MYSQL_USER)
        self.MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', self.MYSQL_PASSWORD)
        self.MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE', self.MYSQL_DATABASE)

        # Redis 配置
        self.REDIS_HOST = os.environ.get('REDIS_HOST', self.REDIS_HOST)
        self.REDIS_PORT = self._env_int('REDIS_PORT', self.REDIS_PORT)
        self.REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', self.REDIS_PASSWORD)
        self.REDIS_DB = self._env_int('REDIS_DB', self.REDIS_DB)

        # Milvus 配置
        self.MILVUS_URI = os.environ.get('MILVUS_URI', self.MILVUS_URI)
        self.MILVUS_HOST = os.environ.get('MILVUS_HOST', self.MILVUS_HOST)
        self.MILVUS_PORT = self._env_int('MILVUS_PORT', self.MILVUS_PORT)
        self.MILVUS_DATABASE_NAME = os.environ.get(
            'MILVUS_DATABASE_NAME', self.MILVUS_DATABASE_NAME
        )
        self.MILVUS_COLLECTION_NAME = os.environ.get(
            'MILVUS_COLLECTION_NAME', self.MILVUS_COLLECTION_NAME
        )

        # LLM 配置（API Key 强烈建议通过环境变量注入）
        self.LLM_MODEL = os.environ.get('LLM_MODEL', self.LLM_MODEL)
        self.DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', self.DASHSCOPE_API_KEY)
        self.DASHSCOPE_BASE_URL = os.environ.get('DASHSCOPE_BASE_URL', self.DASHSCOPE_BASE_URL)

        # 检索参数（支持运行时调整）
        self.PARENT_CHUNK_SIZE = self._env_int('PARENT_CHUNK_SIZE', self.PARENT_CHUNK_SIZE)
        self.CHILD_CHUNK_SIZE = self._env_int('CHILD_CHUNK_SIZE', self.CHILD_CHUNK_SIZE)
        self.CHUNK_OVERLAP = self._env_int('CHUNK_OVERLAP', self.CHUNK_OVERLAP)
        self.RETRIEVAL_K = self._env_int('RETRIEVAL_K', self.RETRIEVAL_K)
        self.CANDIDATE_M = self._env_int('CANDIDATE_M', self.CANDIDATE_M)

        # 应用配置
        valid_sources_env = os.environ.get('VALID_SOURCES')
        if valid_sources_env:
            try:
                self.VALID_SOURCES = ast.literal_eval(valid_sources_env)
            except (ValueError, SyntaxError):
                logger = __import__('logging').getLogger('EduRAG')
                logger.warning(
                    f"环境变量 VALID_SOURCES={valid_sources_env!r} 解析失败，"
                    f"使用 config.ini 回退值"
                )
        self.CUSTOMER_SERVICE_PHONE = os.environ.get(
            'CUSTOMER_SERVICE_PHONE', self.CUSTOMER_SERVICE_PHONE
        )
        self.LOG_FILE = os.environ.get('LOG_FILE', self.LOG_FILE)
