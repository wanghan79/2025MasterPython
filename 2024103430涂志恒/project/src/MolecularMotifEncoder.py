# coding=utf-8
# MolecularMotifEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制（保持与AtomEncoder一致）"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, adj_matrix=None, dist_matrix=None):
        # 计算注意力分数 [batch_size, n_head, seq_len, seq_len]
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        # 处理距离矩阵（确保维度匹配）
        if dist_matrix is not None:
            if dist_matrix.dim() == 3:
                dist_matrix = dist_matrix.unsqueeze(1)
            constant = torch.tensor(1.0, device=dist_matrix.device)
            rescaled_dist = (constant + torch.exp(constant)) / \
                            (constant + torch.exp(constant - dist_matrix))
            attn = F.relu(attn) * rescaled_dist

        # 处理邻接矩阵
        if adj_matrix is not None:
            if adj_matrix.dim() == 3:
                adj_matrix = adj_matrix.unsqueeze(1)
            attn += adj_matrix

        # 应用padding mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """多头注意力机制（与AtomEncoder保持一致）"""

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, f"d_model ({d_model}) must be divisible by n_head ({n_head})"
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, adj_matrix=None, dist_matrix=None):
        residual = q
        batch_size, seq_len = q.size(0), q.size(1)

        # 线性变换并分割头
        q = self.w_qs(q).view(batch_size, seq_len, self.n_head, self.d_k)
        k = self.w_ks(k).view(batch_size, seq_len, self.n_head, self.d_k)
        v = self.w_vs(v).view(batch_size, seq_len, self.n_head, self.d_k)

        # 转置为 [batch_size, n_head, seq_len, d_k]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 计算注意力
        q, attn = self.attention(
            q, k, v,
            mask=mask,
            adj_matrix=adj_matrix,
            dist_matrix=dist_matrix
        )

        # 合并多头输出
        q = q.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络（与AtomEncoder保持一致）"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.gelu(self.w_1(x))
        x = self.dropout(self.w_2(x))
        x += residual
        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """编码器层（简化实现，确保维度流一致）"""

    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None, adj_matrix=None, dist_matrix=None):
        # 多头注意力
        x, attn = self.mha(x, x, x, mask=mask, adj_matrix=adj_matrix, dist_matrix=dist_matrix)
        # 前馈网络
        x = self.ffn(x)
        return x, attn, None  # 保持三个返回值兼容


class MolecularMotifEncoder(nn.Module):
    """基序编码器（确保维度处理与AtomEncoder一致）"""

    def __init__(self, vocab_size, n_layers, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_layers = n_layers

        # 基序嵌入层（+1是为了padding_idx=0）
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # 编码器层堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def create_padding_mask(self, x):
        """创建padding mask（基序ID为0的位置）"""
        mask = (x == 0).float()
        return mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

    def forward(self, x, atom_features=None, adj_matrix=None, dist_matrix=None):
        """
        参数:
            x: 基序ID序列 [batch_size, seq_len]
            atom_features: 原子级特征 [batch_size, seq_len, d_model]（已包含GLOBAL）
            adj_matrix: 邻接矩阵 [batch_size, seq_len, seq_len]
            dist_matrix: 距离矩阵 [batch_size, seq_len, seq_len]
        """
        # 维度检查
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 创建padding mask
        mask = self.create_padding_mask(x)

        # 基序嵌入 [batch_size, seq_len, d_model]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(x)

        # 合并原子级特征（如果提供）
        if atom_features is not None:
            if atom_features.size(1) != x.size(1):  # 处理长度不一致
                atom_features = F.pad(atom_features, (0, 0, 0, x.size(1) - atom_features.size(1)))
            x = x + atom_features

        # 通过编码器层
        attn_weights = []
        for layer in self.encoder_layers:
            x, attn, _ = layer(
                x,
                mask=mask,
                adj_matrix=adj_matrix,
                dist_matrix=dist_matrix
            )
            attn_weights.append(attn)

        return x, attn_weights, None, mask  # 保持与AtomEncoder相同的返回格式


if __name__ == '__main__':
    # 测试配置
    vocab_size = 100  # 假设词汇表大小
    d_model = 256
    n_head = 8

    batch_size = 2
    seq_len = 11  # 包含GLOBAL的motif数量

    # 初始化编码器
    encoder = MolecularMotifEncoder(
        vocab_size=vocab_size,
        n_layers=2,
        d_model=d_model,
        n_head=n_head,
        d_ff=512
    )

    # 模拟输入
    motif_ids = torch.randint(0, vocab_size, (batch_size, seq_len))  # [2,11]
    atom_feat = torch.rand(batch_size, seq_len, d_model)  # 已包含GLOBAL特征
    adj_matrix = torch.rand(batch_size, seq_len, seq_len)
    dist_matrix = torch.rand(batch_size, seq_len, seq_len)

    # 前向传播
    output, attn_weights, _, _ = encoder(
        x=motif_ids,
        atom_features=atom_feat,
        adj_matrix=adj_matrix,
        dist_matrix=dist_matrix
    )

    # 验证输出
    print(f"输入motif_ids形状: {motif_ids.shape}")
    print(f"输入atom_feat形状: {atom_feat.shape}")
    print(f"输出形状: {output.shape}")  # 应该输出 torch.Size([2, 11, 256])
    print(f"注意力权重长度: {len(attn_weights)}")  # 应与n_layers一致

    # 测试padding mask
    # motif_ids[0, -3:] = 0  # 设置padding
    # output, _, _, mask = encoder(motif_ids)
    # print(f"Padding mask形状: {mask.shape}")  # 应该输出 torch.Size([2, 1, 1, 11])
