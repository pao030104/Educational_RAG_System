"""
向量存储与检索模块
===================
基于 Milvus 向量数据库的多模态混合检索引擎，支持：
    - BGE-M3 嵌入：同时生成稠密向量 (Dense) 和稀疏向量 (Sparse)
    - 混合检索 (Hybrid Search)：加权融合稠密和稀疏检索结果
    - BGE-Reranker 重排序：精排检索结果，提升 Top-K 质量
    - 父子块 (Parent-Child Chunk) 策略：子块检索 + 父块返回

集合 Schema 设计:
    ┌─────────────────────────────────────────────────────────────┐
    │ 字段名         │ 类型                │ 描述                   │
    ├─────────────────────────────────────────────────────────────┤
    │ id             │ VARCHAR(100) [PK]   │ 文档内容 MD5 哈希      │
    │ text           │ VARCHAR(65535)      │ 子块文本内容            │
    │ dense_vector   │ FLOAT_VECTOR        │ 稠密向量 (1024维)      │
    │ sparse_vector  │ SPARSE_FLOAT_VECTOR  │ 稀疏向量 (词汇级)      │
    │ parent_id      │ VARCHAR(100)        │ 父块唯一标识            │
    │ parent_content │ VARCHAR(65535)      │ 父块完整内容            │
    │ source         │ VARCHAR(50)         │ 学科类别 (ai/java/...) │
    │ timestamp      │ VARCHAR(50)         │ 文档处理时间戳          │
    └─────────────────────────────────────────────────────────────┘

检索流程:
    查询 → BGE-M3 生成向量 → 混合检索(Dense+Sparse) → 父文档去重 → Reranker精排 → Top-M 结果

模型说明:
    - BGE-M3 (BAAI/bge-m3): 多语言嵌入模型，支持 Dense + Sparse + ColBERT
    - BGE-Reranker-Large (BAAI/bge-reranker-large): Cross-Encoder 重排序模型
"""

# ---- Milvus 向量数据库组件 ----
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker

# ---- BGE-M3 嵌入模型（支持稠密和稀疏双路嵌入） ----
from milvus_model.hybrid import BGEM3EmbeddingFunction

# ---- LangChain Document 类，用于封装检索结果 ----
from langchain.docstore.document import Document

# ---- Cross-Encoder 重排序模型 ----
from sentence_transformers import CrossEncoder

import hashlib                                        # 哈希算法，用于生成文档唯一ID
from base import logger, Config                       # 日志和配置

conf = Config()                                       # 全局配置实例


class VectorStore:
    """
    Milvus 向量存储与混合检索引擎

    封装了向量数据库的完整生命周期管理：
        - 创建/加载集合 (Collection)
        - 文档向量化和插入 (Upsert)
        - 混合检索 (Dense + Sparse)
        - 父文档去重与重排序

    属性:
        collection_name (str):          Milvus 集合名称
        host (str):                     Milvus 服务主机
        port (int):                     Milvus 服务端口
        database (str):                 Milvus 数据库名
        logger:                         日志记录器
        reranker (CrossEncoder):        BGE-Reranker 重排序模型
        embedding_function:             BGE-M3 嵌入函数
        dense_dim (int):                稠密向量维度
        client (MilvusClient):          Milvus 客户端

    使用示例:
        >>> vs = VectorStore()
        >>> docs = process_documents("./data/ai_data")
        >>> vs.add_document(docs)
        >>> results = vs.hybrid_search_with_rerank("什么是AI?", k=5)
    """

    def __init__(
        self,
        collection_name=conf.MILVUS_COLLECTION_NAME,
        host=conf.MILVUS_HOST,
        port=conf.MILVUS_PORT,
        database=conf.MILVUS_DATABASE_NAME,
    ):
        """
        初始化向量存储，依次完成: 加载模型 → 建集合 → 连接 Milvus

        参数:
            collection_name (str): Milvus 集合名称，相当于关系数据库中的"表"
            host (str):           Milvus 服务器地址
            port (int):           Milvus 服务器端口
            database (str):       Milvus 数据库名称
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        self.logger = logger

        # ---- 加载 BGE-Reranker 重排序模型 ----
        # Cross-Encoder 同时编码 Query 和 Document，计算相关性分数
        # 相对于 Bi-Encoder（分别编码后点积），Cross-Encoder 更准确但更慢
        # 使用 HuggingFace Hub 模型名称，首次运行自动下载（约 2.1GB），
        # 缓存于 ~/.cache/huggingface/hub/
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-large",
            device="cpu"
        )

        # ---- 加载 BGE-M3 嵌入模型 ----
        # BGE-M3 同时输出稠密向量 (Dense) 和稀疏向量 (Sparse)
        # use_fp16=False: 使用 FP32 精度（更好的兼容性和精度）
        # device="cpu": CPU 推理（可改为 "cuda" 或 "mps" 加速）
        # 使用 HuggingFace Hub 模型名称，首次运行自动下载（约 2.2GB）。
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name="BAAI/bge-m3",
            use_fp16=False,
            device="cpu"
        )

        # ---- 获取稠密向量的维度 ----
        # BGE-M3 的 Dense 向量维度为 1024
        self.dense_dim = self.embedding_function.dim["dense"]

        # ---- 连接 Milvus 向量数据库 ----
        self.client = MilvusClient(
            uri=f"http://{self.host}:{self.port}",
            db_name=self.database
        )

        # ---- 创建或加载集合 ----
        self._create_or_load_collection()

    def _create_or_load_collection(self):
        """
        创建或加载 Milvus 集合。

        逻辑:
            - 集合不存在 → 定义 Schema + 创建索引 + 创建集合
            - 集合已存在 → 直接加载到内存

        索引类型:
            - 稠密向量: IVF_FLAT (倒排索引 + 平坦搜索)
              params.nlist=128: 聚类中心数为128，平衡速度和精度
            - 稀疏向量: SPARSE_INVERTED_INDEX (倒排索引)
              params.drop_ratio_build=0.2: 构建时丢弃20%低权重项，节省空间
        """
        if not self.client.has_collection(self.collection_name):
            # ---- 定义集合 Schema ----
            # auto_id=False: 使用自定义 ID（文档 MD5 哈希）
            # enable_dynamic_field=True: 允许动态添加未定义的字段
            schema = self.client.create_schema(
                auto_id=False,
                enable_dynamic_field=True
            )

            # 主键字段: 文档内容的 MD5 哈希，用于去重和 Upsert
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=100,
            )
            # 子块文本内容
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=65535,  # VARCHAR 最大长度
            )
            # 稠密向量（1024维浮点向量）
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.dense_dim,
            )
            # 稀疏向量（词汇级稀疏表示）
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            # 父块ID：用于子块→父块的映射
            schema.add_field(
                field_name="parent_id",
                datatype=DataType.VARCHAR,
                max_length=100,
            )
            # 父块内容：完整的上下文文本
            schema.add_field(
                field_name="parent_content",
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
            # 学科类别来源（如 ai, java, bigdata）
            schema.add_field(
                field_name="source",
                datatype=DataType.VARCHAR,
                max_length=50,
            )
            # 文档处理时间戳
            schema.add_field(
                field_name="timestamp",
                datatype=DataType.VARCHAR,
                max_length=50,
            )

            # ---- 创建索引 ----
            index_params = self.client.prepare_index_params()

            # 稠密向量索引: IVF_FLAT
            # metric_type="IP": 内积 (Inner Product) 作为相似度度量
            #   - BGE 嵌入已做 L2 归一化，因此内积等价于余弦相似度
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128},
            )

            # 稀疏向量索引: SPARSE_INVERTED_INDEX
            # 使用倒排索引加速稀疏向量的近似最近邻搜索
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )

            # 创建集合
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"已创建集合 {self.collection_name}")
        else:
            logger.info(f"已加载集合 {self.collection_name}")

        # 将集合加载到内存，确保立即可查询
        # 不加载的话，集合在首次查询时自动加载（有冷启动延迟）
        self.client.load_collection(self.collection_name)

    def add_document(self, documents):
        """
        将文档块批量插入（或更新）到 Milvus 集合。

        使用 Upsert 语义:
            - id 不存在 → 插入新记录
            - id 已存在 → 更新已有记录（适合增量更新场景）

        处理流程:
            1. 提取所有文档的文本内容
            2. 使用 BGE-M3 生成 Dense + Sparse 向量
            3. 为每个文档生成 MD5 唯一 ID
            4. 构建包含所有字段的数据记录
            5. 批量 Upsert 到 Milvus

        参数:
            documents (list[Document]): LangChain Document 对象列表，
                                        每个 Document 需包含:
                                        - page_content: 子块文本
                                        - metadata.parent_id: 父块ID
                                        - metadata.parent_content: 父块内容
                                        - metadata.source: 学科类别
                                        - metadata.timestamp: 时间戳
        """
        # 提取所有文档的文本内容
        texts = [doc.page_content for doc in documents]

        # BGE-M3 生成嵌入向量
        # embeddings["dense"]: 稠密向量矩阵 [N, 1024]
        # embeddings["sparse"]: 稀疏向量矩阵 (CSR格式)
        embeddings = self.embedding_function(texts)

        data = []  # 待插入的数据记录列表
        for i, doc in enumerate(documents):
            # ---- 生成唯一ID ----
            # 使用 MD5 哈希：相同内容 → 相同ID → Upsert 自动去重
            text_hash = hashlib.md5(
                doc.page_content.encode("utf-8")
            ).hexdigest()

            # ---- 构建稀疏向量字典 ----
            # Milvus 要求稀疏向量以 dict[int, float] 格式存储
            # 从 CSR 矩阵的第 i 行提取非零元素
            sparse_vector = {}
            row = embeddings["sparse"].getrow(i)   # 获取第 i 行
            indices = row.indices                    # 非零值的列索引
            values = row.data                        # 非零值本身
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value           # 填充字典

            # ---- 构建完整的数据记录 ----
            data.append({
                "id": text_hash,                              # MD5 哈希主键
                "text": doc.page_content,                     # 子块文本
                "dense_vector": embeddings["dense"][i],       # 稠密向量
                "sparse_vector": sparse_vector,               # 稀疏向量
                "parent_id": doc.metadata["parent_id"],       # 父块ID
                "parent_content": doc.metadata["parent_content"],  # 父块内容
                "source": doc.metadata.get("source", "unknown"),    # 学科类别
                "timestamp": doc.metadata.get("timestamp", "unknown"), # 时间戳
            })

        # 批量 Upsert（有则更新，无则插入）
        if data:
            self.client.upsert(
                collection_name=self.collection_name,
                data=data
            )
            logger.info(f"已插入或更新{len(data)}个文档")

    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        """
        执行完整的混合检索 + 重排序流程。

        这是本系统的核心检索方法，包含以下步骤:

        ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
        │ 用户查询      │───→│ BGE-M3 生成向量    │───→│ 稠密向量检索     │
        │              │    │ (Dense + Sparse) │    │ + 稀疏向量检索   │
        └──────────────┘    └──────────────────┘    └─────────────────┘
                                                            │
                                                            ▼
        ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
        │ Top-M 结果   │←───│ BGE-Reranker     │←───│ 父文档去重      │
        │              │    │ 精排              │    │                 │
        └──────────────┘    └──────────────────┘    └─────────────────┘

        参数:
            query (str): 用户查询文本。
            k (int): 混合检索阶段的召回数量，默认取配置中的 RETRIEVAL_K。
            source_filter (str, optional): 学科类别过滤。为 None 时不过滤。

        返回:
            list[Document]: 精排后的父文档列表，最多返回 CANDIDATE_M 个。

        注意:
            - 混合检索权重: 稀疏 0.7 + 稠密 1.0 = 共 1.7
            - 稀疏权重较高是为了侧重关键词匹配，适合专业知识检索
            - 若父文档数 < 2，跳过重排序直接返回
        """
        # ---- 步骤1: 生成查询的嵌入向量 ----
        query_embeddings = self.embedding_function([query])

        # 提取稠密查询向量 (numpy array shape=[1024])
        dense_query_vector = query_embeddings["dense"][0]

        # 提取稀疏查询向量 (dict[int, float])
        sparse_query_vector = {}
        row = query_embeddings["sparse"].getrow(0)
        for idx, value in zip(row.indices, row.data):
            sparse_query_vector[idx] = value

        # ---- 步骤2: 构造过滤表达式 ----
        # 用于按学科类别过滤（如 source=="ai"），空字符串表示不过滤
        filter_expr = f'source=="{source_filter}"' if source_filter else ""

        # ---- 步骤3: 构造稠密向量搜索请求 ----
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",          # 搜索字段
            param={"metric_type": "IP", "params": {}},  # 内积度量
            limit=k,                            # 返回 top-k
            expr=filter_expr,                   # 过滤表达式
        )

        # ---- 步骤4: 构造稀疏向量搜索请求 ----
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr,
        )

        # ---- 步骤5: 混合检索（加权融合） ----
        # WeightedRanker 对两路搜索结果进行加权求和排序
        # 权重: 稀疏=0.7, 稠密=1.0 (稀疏侧重关键词匹配)
        ranker = WeightedRanker(0.7, 1.0)

        # hybrid_search 返回 List[List[dict]]，取 [0] 获取第一个查询的结果
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[sparse_request, dense_request],  # 注意: 先稀疏后稠密，与 WeightedRanker 参数顺序对应
            ranker=ranker,
            limit=k,
            output_fields=[                        # 指定返回的字段
                "text",
                "parent_id",
                "parent_content",
                "source",
                "timestamp",
            ],
        )[0]  # 取第一个（也是唯一一个）查询的结果

        # ---- 步骤6: 将搜索结果转为 Document 对象 ----
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        self.logger.debug(f"混合检索返回 {len(sub_chunks)} 个子块")

        # ---- 步骤7: 父文档去重 ----
        # 从子块中提取唯一的父文档（同一父块的多个子块只保留一个父文档）
        parent_docs = self._get_unique_parent_docs(sub_chunks)

        # ---- 步骤8: 重排序（可选） ----
        # 仅当父文档数量 ≥ 2 时进行重排序（1个文档排序无意义）
        if len(parent_docs) < 2:
            return parent_docs[:conf.CANDIDATE_M]

        if parent_docs:
            # 构造 (query, document) 配对列表
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # Cross-Encoder 对每对计算相关性分数
            scores = self.reranker.predict(pairs)
            # 按分数降序排列文档
            ranked_parent_docs = [
                doc for _, doc in sorted(
                    zip(scores, parent_docs),
                    key=lambda x: x[0],  # 按分数排序
                    reverse=True          # 降序
                )
            ]
        else:
            ranked_parent_docs = []

        # ---- 步骤9: 返回 Top-M 结果 ----
        # CANDIDATE_M 为最终保留的文档数（配置中定义）
        return ranked_parent_docs[:conf.CANDIDATE_M]

    def _get_unique_parent_docs(self, sub_chunks):
        """
        从子块列表中提取去重的父文档。

        原理:
            父子块策略中，一个父块可能被切分成多个子块。
            在检索阶段，多个子块可能命中同一个父块。
            此方法将子块映射回父文档并去重，确保返回的上下文不重复。

        参数:
            sub_chunks (list[Document]): 混合检索返回的子块 Document 列表。

        返回:
            list[Document]: 去重后的父文档列表。
        """
        parent_contents = set()   # 已处理的父块内容哈希集，用于去重
        unique_docs = []          # 去重后的父文档列表

        for chunk in sub_chunks:
            # 从子块元数据中提取父块内容，若没有则用子块内容作为回退
            parent_content = chunk.metadata.get(
                'parent_content', chunk.page_content
            )
            # 只有非空且未处理过的内容才添加
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，内容为父块内容，保留原始元数据
                unique_docs.append(
                    Document(
                        page_content=parent_content,
                        metadata=chunk.metadata
                    )
                )
                parent_contents.add(parent_content)

        return unique_docs

    def _doc_from_hit(self, hit):
        """
        将 Milvus 搜索结果中的一条命中记录转换为 LangChain Document 对象。

        参数:
            hit (dict): Milvus 的一条搜索结果，包含 text 和元数据字段。

        返回:
            Document: LangChain Document 对象。
        """
        return Document(
            page_content=hit.get('text'),     # 子块文本内容
            metadata={
                'parent_id': hit.get('parent_id'),           # 父块ID
                'parent_content': hit.get('parent_content'),  # 父块完整内容
                'source': hit.get('source'),                  # 学科类别
                'timestamp': hit.get('timestamp')             # 时间戳
            }
        )
