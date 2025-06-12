# model/transformer_model.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model//2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_encoder = PositionalEncoding(config.embedding_dim, config.max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.fc = nn.Linear(config.embedding_dim, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = self.pos_encoder(embedded)
        x = self.transformer_encoder(x)  # [batch_size, seq_len, embedding_dim]
        x = x[:, 0, :]  # 取CLS位，或平均池化
        x = self.dropout(x)
        out = self.fc(x)
        return out

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    layerwise = {name: p.numel() for name, p in model.named_parameters()}
    return total, layerwise
