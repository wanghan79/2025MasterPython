# coding=utf-8
# MolecularAtomEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, adj_matrix=None, dist_matrix=None):
        # 计算注意力分数 [batch_size, n_head, seq_len, seq_len]
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        # # 处理距离矩阵（关键修复：确保维度匹配）
        # if dist_matrix is not None:
        #     # 确保dist_matrix已经多头扩展
        #     if dist_matrix.dim() == 3:  # 原始输入 [batch_size, seq_len, seq_len]
        #         dist_matrix = dist_matrix.unsqueeze(1)  # 添加头维度
        #     constant = torch.tensor(1.0, device=dist_matrix.device)
        #     rescaled_dist = (constant + torch.exp(constant)) / \
        #                     (constant + torch.exp(constant - dist_matrix))
        #     attn = F.relu(attn) * rescaled_dist
        #
        # # 处理邻接矩阵（同样需要多头扩展）
        # if adj_matrix is not None:
        #     if adj_matrix.dim() == 3:
        #         adj_matrix = adj_matrix.unsqueeze(1)
        #     attn += adj_matrix

        # 应用padding mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        outputs = torch.matmul(attn, v)
        return outputs, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_model = d_model

        # 线性变换层
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, adj_matrix=None, dist_matrix=None):
        residual = q  # [batch_size, seq_len, d_model]
        batch_size, seq_len = q.size(0), q.size(1)

        # 线性变换并分割头
        q = self.w_qs(q).view(batch_size, seq_len, self.n_head, self.d_k)
        k = self.w_ks(k).view(batch_size, seq_len, self.n_head, self.d_k)
        v = self.w_vs(v).view(batch_size, seq_len, self.n_head, self.d_k)

        # 转置为 [batch_size, n_head, seq_len, d_k]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 计算注意力
        q, attn = self.attention(q, k, v, mask=mask, adj_matrix=adj_matrix, dist_matrix=dist_matrix)

        # 合并多头输出
        q = q.transpose(1, 2).contiguous()  # [batch_size, seq_len, n_head, d_k]
        q = q.view(batch_size, seq_len, -1)  # 使用-1自动计算d_model维度

        # 最终线性变换
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        # 确保输入输出维度一致
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)  # 输出维度恢复为d_in
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
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None, adj_matrix=None, dist_matrix=None):
        # 多头注意力
        x, attn = self.mha(x, x, x, mask=mask, adj_matrix=adj_matrix, dist_matrix=dist_matrix)
        # 前馈网络
        x = self.ffn(x)
        return x, attn


def create_padding_mask(x):
    """创建padding mask（原子特征全零的位置）"""
    mask = (torch.sum(x, dim=-1) == 0).float()
    return mask.unsqueeze(1).unsqueeze(2)


class MolecularAtomEncoder(nn.Module):
    """原子编码器（关键修改：确保维度一致性）"""

    def __init__(self, n_layers, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model

        self.atom_embedding = nn.Sequential(
            nn.Linear(67, d_model),  # 输入维度67
            nn.ReLU()
        )

        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, adj_matrix=None, dist_matrix=None, mask_matrix=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            if adj_matrix is not None:
                adj_matrix = adj_matrix.unsqueeze(0)
            if dist_matrix is not None:
                dist_matrix = dist_matrix.unsqueeze(0)

        mask = mask_matrix.unsqueeze(1).unsqueeze(2) if mask_matrix is not None else create_padding_mask(x)

        # 原子特征嵌入 [batch_size, seq_len, d_model]
        x = self.atom_embedding(x)
        x = self.dropout(x)

        # 通过编码器层
        attn_weights = []
        for layer in self.encoder_layers:
            x, attn = layer(
                x,
                mask=mask,
                adj_matrix=adj_matrix,
                dist_matrix=dist_matrix
            )
            attn_weights.append(attn)

        if mask_matrix is not None:
            mask = mask_matrix.unsqueeze(-1)  # [batch, seq_len, 1]
            sum_features = torch.sum(x * mask, dim=1)
            valid_counts = torch.sum(mask, dim=1)
            global_feature = sum_features / (valid_counts + 1e-9)
        else:
            global_feature = torch.mean(x, dim=1)

        return x, global_feature


if __name__ == '__main__':
    encoder = MolecularAtomEncoder(n_layers=2, d_model=256, n_head=8, d_ff=512)
    output, gl_feat = encoder(
        torch.rand(5, 33, 67),
        adj_matrix=torch.rand(5, 33, 33),
        dist_matrix=torch.rand(5, 33, 33),
        mask_matrix=torch.tensor([[1] * 30 + [0] * 3] * 5)
    )
    print('输出形状:', output.shape, gl_feat.shape)
