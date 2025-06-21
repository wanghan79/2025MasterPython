import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 检查 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# 加载数据集
df = pd.read_csv('./dataset/cora_text_dataset.csv')
# 本地模型路径
model_path = './model/bert-base-uncased'
# 初始化 BERT 模型和分词器（从本地加载）
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True)
model.to(device)  # 将模型移动到 GPU
# 将 title 和 abstract 结合
def combine_title_abstract(row):
    return f"{row['Title']} {row['Abstract']}"
df['combined_text'] = df.apply(combine_title_abstract, axis=1)
# 批量提取特征
def extract_features_batch(texts):
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 将输入移动到 GPU
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用 [CLS] token 的向量作为特征
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # 将结果移回 CPU
# 分批处理
batch_size = 32
combined_texts = df['combined_text'].tolist()
node_features = []
for i in range(0, len(combined_texts), batch_size):
    batch_texts = combined_texts[i:i + batch_size]
    batch_features = extract_features_batch(batch_texts)
    node_features.extend(batch_features)
# 保存特征到 DataFrame
df['node_feat'] = [' '.join(map(str, feature)) for feature in node_features]
# 创建新的 DataFrame
output_df = df[['PaperID', 'node_feat']]
# 保存到 CSV 文件
output_df.to_csv('./dataset/cora_feat768.csv', index=False)