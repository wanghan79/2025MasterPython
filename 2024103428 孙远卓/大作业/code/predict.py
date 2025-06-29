import torch
from model import SentimentLSTM, initialize_model
from data_loader import load_vocab, clean_text
import config
import json


class SentimentPredictor:
    def __init__(self):
        cfg = config.Config()

        # 加载词汇表
        self.word2idx = load_vocab(cfg.vocab_save_path)
        vocab_size = len(self.word2idx)

        # 加载模型
        self.model = initialize_model(vocab_size)
        self.model.load_state_dict(torch.load(cfg.model_save_path, map_location=torch.device(cfg.device)))
        self.model = self.model.to(cfg.device)
        self.model.eval()

        self.max_len = cfg.max_len

    def predict(self, text):
        """预测评论情感"""
        tokens = clean_text(text)
        indexed = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]

        # 截断或填充序列
        if len(indexed) < self.max_len:
            indexed += [self.word2idx['<pad>']] * (self.max_len - len(indexed))
        else:
            indexed = indexed[:self.max_len]

        # 转换为tensor
        tensor = torch.tensor(indexed).unsqueeze(0).to(config.Config.device)

        # 预测
        with torch.no_grad():
            output = self.model(tensor).squeeze(1)
            proba = torch.sigmoid(output).item()
            sentiment = "positive" if proba > 0.5 else "negative"

        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": proba if sentiment == "positive" else 1 - proba,
            "positive_probability": proba
        }

    def batch_predict(self, texts):
        """批量预测评论情感"""
        return [self.predict(text) for text in texts]


def save_predictions(predictions, filename):
    """保存预测结果到文件"""
    with open(filename, 'w') as f:
        json.dump(predictions, f, indent=2)