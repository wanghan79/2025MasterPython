import torch
from model import SentimentLSTM, initialize_model
from data_loader import load_vocab, clean_text  # 添加 clean_text 导入
import config
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# 添加 load_and_preprocess_data 函数
def load_and_preprocess_data():
    """加载和预处理数据 - 用于评估的简化版本"""
    from data_loader import load_and_preprocess_data as original_load_data
    return original_load_data()


def evaluate_trained_model():
    """评估已训练模型"""
    cfg = config.Config()

    # 加载词汇表
    word2idx = load_vocab(cfg.vocab_save_path)
    vocab_size = len(word2idx)

    # 加载模型
    model = initialize_model(vocab_size)
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=torch.device(cfg.device)))
    model = model.to(cfg.device)
    model.eval()

    # 加载测试数据
    _, test_loader, _, _ = load_and_preprocess_data()

    # 评估模型
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = batch
            inputs = inputs.to(cfg.device)
            outputs = model(inputs).squeeze(1)
            preds = torch.round(torch.sigmoid(outputs))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['negative', 'positive']))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['negative', 'positive'],
                yticklabels=['negative', 'positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')
    plt.show()

    return all_preds, all_labels


def predict_sentiment(model, word2idx, text, max_len):
    """预测单条评论的情感"""
    tokens = clean_text(text)
    indexed = [word2idx.get(token, word2idx['<unk>']) for token in tokens]

    # 截断或填充序列
    if len(indexed) < max_len:
        indexed += [word2idx['<pad>']] * (max_len - len(indexed))
    else:
        indexed = indexed[:max_len]

    # 转换为tensor
    tensor = torch.tensor(indexed).unsqueeze(0).to(config.Config.device)

    # 预测
    with torch.no_grad():
        output = model(tensor).squeeze(1)
        proba = torch.sigmoid(output).item()
        sentiment = "positive" if proba > 0.5 else "negative"

    return sentiment, proba


# 添加独立运行支持
if __name__ == "__main__":
    evaluate_trained_model()