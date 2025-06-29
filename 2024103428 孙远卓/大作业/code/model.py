import torch
import torch.nn as nn
import config


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,  # 双向LSTM
            batch_first=True,
            dropout=dropout
        )

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 双向所以是2倍

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text: [batch_size, seq_len]

        # 词嵌入
        embedded = self.dropout(self.embedding(text))  # [batch_size, seq_len, emb_dim]

        # LSTM处理
        output, (hidden, cell) = self.lstm(embedded)

        # 取最后一个时间步的输出
        # 双向LSTM，需要连接最后两个隐藏状态
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # 全连接层
        return self.fc(hidden)


def initialize_model(vocab_size):
    """初始化模型"""
    cfg = config.Config()
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim,
        dropout=cfg.dropout
    )
    return model


def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)