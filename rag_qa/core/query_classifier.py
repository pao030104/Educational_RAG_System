"""
查询分类器模块
===============
使用微调后的 BERT 模型对用户查询进行二分类：通用知识 vs 专业咨询。

分类逻辑:
    ┌─────────────────────────────────────────────────────┐
    │ 用户查询                                              │
    │   ↓                                                  │
    │ BERT 分类器                                          │
    │   ├── "通用知识" (label=0): 不检索，直接用 LLM 回答     │
    │   └── "专业咨询" (label=1): 触发 RAG 检索流程          │
    └─────────────────────────────────────────────────────┘

为什么要做查询分类？
    1. 避免浪费：简单通用问题不需要检索知识库
    2. 提升速度：跳过检索步骤，通用问题秒回
    3. 改善体验：专业知识问题能获取更精准的上下文

模型架构:
    - 基座模型: bert-base-chinese (中文 BERT)
    - 微调任务: 序列分类 (Sequence Classification)
    - 类别数: 2 (通用知识 = 0, 专业咨询 = 1)
    - 输入长度: 最大 128 tokens

训练数据:
    - classify_data/model_generic_5000.json (5000 条标注数据)
    - JSONL 格式，每行: {"query": "问题", "label": "通用知识"|"专业咨询"}
"""

import sys
import os

# ---- 设置 Python 模块搜索路径 ----
current_dir = os.path.dirname(os.path.abspath(__file__))       # core/
rag_qa_path = os.path.dirname(current_dir)                      # rag_qa/
project_root = os.path.dirname(rag_qa_path)                     # 项目根目录
sys.path.insert(0, project_root)

import json                                      # JSON 数据处理
import shutil                                    # 文件和目录操作，用于清理临时文件

import torch                                     # PyTorch 深度学习框架
from base import logger                          # 全局日志器
import numpy as np                               # 数值计算库

# ---- Transformers 库组件 ----
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

# ---- Scikit-learn 评估工具 ----
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class QueryClassifier:
    """
    基于 BERT 的查询意图分类器

    加载微调后的 BERT 模型，对用户查询进行二分类，判断是否需要触发 RAG 检索。

    属性:
        model_path (str):  微调后的 BERT 模型保存路径
        tokenizer:         BERT 分词器 (BertTokenizer)
        model:             BERT 分类模型 (BertForSequenceClassification)
        device:            推理设备 (MPS/CPU)
        label_map (dict):  标签映射 {"通用知识": 0, "专业咨询": 1}

    使用示例:
        >>> classifier = QueryClassifier()
        >>> category = classifier.predict_category("AI学科学费多少？")
        >>> print(category)  # 输出: "专业咨询"
        >>> category = classifier.predict_category("今天天气怎么样？")
        >>> print(category)  # 输出: "通用知识"
    """

    def __init__(self, model_path=None):
        """
        初始化查询分类器。

        处理流程:
            1. 确定模型路径（默认: rag_qa/core/bert_query_classifier/）
            2. 加载 BERT 分词器（从本地 bert-base-chinese 模型）
            3. 确定推理设备（Apple Silicon → MPS，否则 → CPU）
            4. 加载或初始化分类模型
            5. 定义标签映射

        参数:
            model_path (str, optional): 微调后模型的路径。
                                       默认使用 core/bert_query_classifier/。
        """
        # ---- 确定模型保存路径 ----
        if model_path is None:
            # 默认保存到 core 目录下的 bert_query_classifier 子目录
            self.model_path = os.path.join(current_dir, "bert_query_classifier")
        else:
            self.model_path = model_path

        # ---- 加载 BERT 分词器 ----
        # 使用 HuggingFace Hub 上的 google-bert/bert-base-chinese 模型，
        # 首次运行会自动下载（约 400MB），缓存于 ~/.cache/huggingface/hub/
        self.tokenizer = BertTokenizer.from_pretrained(
            "google-bert/bert-base-chinese"
        )

        # ---- 初始化模型（稍后在 load_model 中加载权重） ----
        self.model = None

        # ---- 确定推理设备 ----
        # Apple Silicon Mac 优先使用 MPS (Metal Performance Shaders) 加速
        # CUDA GPU 次之，其他设备使用 CPU
        # 使用 getattr 兼容旧版 PyTorch（无 mps 属性）
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info("推理设备: %s" % self.device)

        # ---- 定义标签映射 ----
        # 二分类: 通用知识=0, 专业咨询=1
        # 预测时: prediction=0 → "通用知识", prediction=1 → "专业咨询"
        self.label_map = {"通用知识": 0, "专业咨询": 1}

        # ---- 加载模型权重 ----
        self.load_model()

    def load_model(self):
        """
        加载 BERT 分类模型。

        加载优先级:
            1. 如果 model_path 下存在微调后的模型（含 model.safetensors）→ 直接加载
            2. 否则 → 从 HuggingFace 下载 bert-base-chinese 基座，并自动训练微调模型
               （训练数据来自 classify_data/model_generic_5000.json）

        加载完成后自动将模型移动到目标设备（MPS 或 CPU）。

        注意:
            首次运行时若无已微调的模型，会自动触发训练（约需 10-20 分钟，CPU），
            训练完成后模型保存在 model_path 下，后续启动将直接加载。
        """
        # 检查微调后的模型是否存在（至少需要 config.json）
        finetuned_config = os.path.join(self.model_path, "config.json")
        if os.path.exists(finetuned_config):
            # 加载已微调的模型
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_path
            )
            self.model.to(self.device)  # 移动到目标设备
            logger.info(f"加载微调模型成功: {self.model_path}")
        else:
            # 从 HuggingFace Hub 下载 bert-base-chinese 基座模型
            # 首次运行自动下载（约 400MB），缓存于 ~/.cache/huggingface/hub/
            logger.info("未找到微调模型，从 HuggingFace 下载 bert-base-chinese 基座...")
            self.model = BertForSequenceClassification.from_pretrained(
                "google-bert/bert-base-chinese",
                num_labels=len(self.label_map)  # num_labels=2
            )
            self.model.to(self.device)
            logger.info("基座模型加载成功，开始自动训练微调模型...")
            # 自动训练微调模型（使用内置训练数据）
            self.train_model(data_file="classify_data/model_generic_5000.json")

    def save_model(self):
        """
        保存微调后的模型和分词器到 model_path。

        同时保存:
            - 模型权重 (model.safetensors)
            - 模型配置 (config.json)
            - 分词器配置 (tokenizer_config.json, vocab.txt)
        """
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"保存模型成功:{self.model_path}")

    def preprocess_data(self, text, labels):
        """
        对文本数据和标签进行预处理（分词 + 标签编码）。

        处理步骤:
            1. 使用 BERT Tokenizer 进行分词、截断、填充
            2. 将文本标签映射为整数 (通用知识→0, 专业咨询→1)

        参数:
            text (list[str]): 文本列表
            labels (list[str]): 标签列表（如 ["通用知识", "专业咨询", ...]）

        返回:
            tuple: (encodings, encoded_labels)
                - encodings: transformers 的 BatchEncoding 对象
                - encoded_labels: 整数标签列表
        """
        # 分词处理: 截断超过128的文本，填充到最长文本长度
        encodings = self.tokenizer(
            text,
            truncation=True,    # 超长截断
            padding=True,       # 短文本填充到批次最长
            max_length=128,     # 最大长度限制
            return_tensors="pt" # 返回 PyTorch 张量
        )
        # 将字符串标签转换为整数
        return encodings, [self.label_map[label] for label in labels]

    def create_dataset(self, encodings, labels):
        """
        创建 PyTorch Dataset 对象，用于 Trainer 的训练和评估。

        参数:
            encodings: Tokenizer 的输出（BatchEncoding）
            labels (list[int]): 整数标签

        返回:
            torch.utils.data.Dataset: 可直接输入 Trainer 的数据集
        """
        # 定义内部 Dataset 类，遵循 PyTorch Dataset 协议
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                # 返回第 idx 个样本的编码 + 标签
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return Dataset(encodings, labels)

    def train_model(self, data_file="classify_data/model_generic_5000.json"):
        """
        使用标注数据训练 BERT 分类模型。

        训练流程:
            1. 加载 JSONL 格式的训练数据
            2. 按 80/20 比例划分训练集和验证集
            3. 数据预处理（分词 + 标签编码）
            4. 配置训练参数（3 epoch, batch_size=8, 学习率调度）
            5. 执行训练（自动保存最佳模型）
            6. 在验证集上评估
            7. 清理训练产生的临时文件

        参数:
            data_file (str): JSONL 格式的训练数据文件路径。

        异常:
            FileNotFoundError: 训练数据文件不存在时抛出。
        """
        # ---- 解析训练数据文件路径 ----
        if not os.path.isabs(data_file):
            # 相对路径优先从 rag_qa 目录查找
            resolved_path = os.path.join(rag_qa_path, data_file)
            if os.path.exists(resolved_path):
                data_file = resolved_path

        if not os.path.exists(data_file):
            logger.error(f"训练数据文件 {data_file} 不存在")
            raise FileNotFoundError(f"训练数据文件 {data_file} 不存在")

        # ---- 加载 JSONL 数据 ----
        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        # 提取文本和标签
        texts = [item["query"] for item in data]
        labels = [item["label"] for item in data]

        # ---- 数据划分: 80% 训练 / 20% 验证 ----
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=3
        )

        # ---- 数据预处理 ----
        train_encodings, train_labels = self.preprocess_data(
            train_texts, train_labels
        )
        eval_encodings, val_labels = self.preprocess_data(
            val_texts, val_labels
        )

        # ---- 创建 PyTorch Dataset ----
        train_dataset = self.create_dataset(train_encodings, train_labels)
        eval_dataset = self.create_dataset(eval_encodings, val_labels)

        # ---- 设置训练参数 ----
        bert_results_dir = os.path.join(current_dir, "bert_results")     # 模型保存目录
        bert_logs_dir = os.path.join(current_dir, "bert_logs")          # TensorBoard 日志目录

        training_args = TrainingArguments(
            output_dir=bert_results_dir,                   # 输出目录
            num_train_epochs=3,                            # 训练轮数
            per_device_train_batch_size=8,                 # 训练批次大小
            per_device_eval_batch_size=8,                  # 评估批次大小
            warmup_steps=500,                              # 学习率预热步数
            weight_decay=0.01,                             # 权重衰减（正则化）
            logging_dir=bert_logs_dir,                     # 日志目录
            logging_steps=10,                              # 每10步记录一次日志
            eval_strategy="epoch",                         # 每个 epoch 结束时评估
            save_strategy="epoch",                         # 每个 epoch 结束时保存
            load_best_model_at_end=True,                   # 训练结束装载最佳模型
            save_total_limit=1,                            # 只保留1个最佳模型
            metric_for_best_model="eval_loss",             # 以验证损失选择最佳模型
            fp16=False,                                    # 不使用混合精度（MPS 不支持）
        )

        # ---- 初始化 Trainer ----
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,  # 自定义评估指标函数
        )

        # ---- 执行训练 ----
        logger.info("开始训练BERT模型...")
        trainer.train()

        # ---- 保存模型 ----
        # 训练后将模型移回 CPU 再保存，确保兼容性
        self.model = self.model.to("cpu")
        self.save_model()
        self.load_model()  # 重新加载以获取干净的状态

        # ---- 在验证集上评估 ----
        bert_eval_dir = os.path.join(current_dir, "bert_eval_tmp")
        self.evaluate_model(val_texts, val_labels, bert_eval_dir)

        # ---- 清理临时文件夹 ----
        for tmp_dir in [bert_results_dir, bert_logs_dir, bert_eval_dir]:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
                logger.info(f"已清理临时目录: {tmp_dir}")

    def compute_metrics(self, eval_pred):
        """
        计算评估指标（准确率）。

        参数:
            eval_pred: Trainer 传入的 EvalPrediction 对象，包含 logits 和 labels。

        返回:
            dict: 指标字典，如 {"accuracy": 0.95}。
        """
        logits, labels = eval_pred
        # argmax 获取预测类别，axis=-1 表示沿最后一个维度
        predictions = np.argmax(logits, axis=-1)
        # 计算准确率: 预测正确的比例
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def evaluate_model(self, texts, labels, eval_output_dir):
        """
        在给定测试集上评估模型，输出分类报告和混淆矩阵。

        参数:
            texts (list[str]): 测试文本列表
            labels (list[int]): 真实标签列表（整数）
            eval_output_dir (str): 评估临时输出目录
        """
        # 分词处理
        encodings = self.tokenizer(
            texts, truncation=True, padding=True,
            max_length=128, return_tensors="pt"
        )
        dataset = self.create_dataset(encodings, labels)

        # 创建评估用的 Trainer
        eval_args = TrainingArguments(
            output_dir=eval_output_dir,
            per_device_eval_batch_size=8,
        )
        trainer = Trainer(model=self.model, args=eval_args)
        predictions = trainer.predict(dataset)

        # 预测完成后将模型移回目标设备
        self.model = self.model.to(self.device)

        # 获取预测标签
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = labels

        # 输出详细的分类报告（精确率、召回率、F1）
        logger.info("分类报告:")
        logger.info(classification_report(
            true_labels, pred_labels,
            target_names=["通用知识", "专业咨询"]
        ))
        # 输出混淆矩阵（直观展示分类错误分布）
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels))

    def predict_category(self, query):
        """
        对单个查询进行意图分类。

        推理流程:
            1. 检查模型是否已加载
            2. 切换到 eval 模式（关闭 Dropout）
            3. 分词 + 移动到设备
            4. 前向传播（无梯度计算）
            5. argmax 获取预测类别

        参数:
            query (str): 用户查询文本。

        返回:
            str: "通用知识" 或 "专业咨询"。

        使用示例:
            >>> classifier = QueryClassifier()
            >>> result = classifier.predict_category("5*9等于多少?")
            >>> print(result)  # "通用知识"
        """
        # 安全检查：模型未加载时返回默认类别
        if self.model is None:
            logger.error("模型未加载")
            return '通用知识'

        # 切换到评估模式：关闭 Dropout 层，保证推理确定性
        self.model.eval()
        # 确保模型在正确的设备上
        self.model = self.model.to(self.device)

        # 分词处理
        encoding = self.tokenizer(
            query,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        # 将编码张量移动到目标设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # 推理（无梯度，节省显存和加速）
        with torch.no_grad():
            outputs = self.model(**encoding)
            # argmax 获取概率最大的类别索引
            prediction = torch.argmax(outputs.logits, dim=1).item()

        # 将整数索引映射回标签字符串
        return '专业咨询' if prediction == 1 else '通用知识'


if __name__ == '__main__':
    # ===== 模型训练和使用演示 =====
    classifier = QueryClassifier()

    # 训练模型（如已有微调模型可跳过）
    classifier.train_model(data_file='classify_data/model_generic_5000.json')

    # 测试查询分类
    test_query = [
        'AI学科的课程大纲是什么',      # 专业咨询（涉及特定学科知识库）
        '如何评价AI学科的课程',         # 专业咨询
        '5*9等于多少?'                 # 通用知识（基础算术）
    ]
    for query in test_query:
        category = classifier.predict_category(query)
        print(f'{query}的类别是{category}')
