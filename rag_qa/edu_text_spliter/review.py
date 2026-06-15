# class A():
#     def a1(self):
#         print('你好A')
#     def b1(self):
#         self.a1()
#
# class B(A):
#     def a1(self,):
#         print('你好B')
#
# b = B()
# b.b1()


import sys, os
# 计算项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(os.path.dirname(current_dir))  # edu_text_spliter -> rag_qa
sys.path.insert(0, rag_qa_path)

from transformers import BertModel
from modelscope import snapshot_download

# 使用 ModelScope 下载模型（首次运行自动下载，约 400MB）
model_dir = snapshot_download(
    'iic/nlp_bert_document-segmentation_chinese-base'
)
bert_model = BertModel.from_pretrained(model_dir)
print(bert_model)

