# coding=utf-8
# ProteinGNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch
import pickle
from pathlib import Path
from torch_geometric.data import Data
# from torch_scatter import scatter_mean
from abc import ABC


class GeoGCNConv(MessagePassing, ABC):
    """几何感知的图卷积层"""

    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='mean')
        self.edge_mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, in_dim)
        )
        self.node_mlp = nn.Linear(in_dim, out_dim)  # 共享权重机制
        self.edge_proj = nn.Linear(in_dim, out_dim)  # 边权重投影

    def forward(self, x, edge_index, edge_attr):
        # 计算边权重 [E, in_dim]
        edge_weights = self.edge_mlp(edge_attr)
        # 共享权重消息传递
        return self.propagate(edge_index, x=x, edge_weights=edge_weights)

    def message(self, x_j, edge_weights):
        return self.edge_proj(x_j * edge_weights)  # 融合节点特征与边权重 [E, out_dim]

    def update(self, aggr_out, x):
        return self.node_mlp(x) + aggr_out  # 残差连接


class ProteinGNN(nn.Module):
    def __init__(self, node_dim=24, hidden_dim=128, out_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        self.conv1 = GeoGCNConv(node_dim + 32, hidden_dim)
        self.conv2 = GeoGCNConv(hidden_dim, hidden_dim)
        self.conv3 = GeoGCNConv(hidden_dim, out_dim)
        # 批量归一化
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, data):
        invariants = torch.stack([
            torch.norm(data.pos, dim=1),  # 到原点距离
            data.pos[:, 0] ** 2 + data.pos[:, 1] ** 2,  # 柱坐标分量
            data.pos[:, 0] * data.pos[:, 1]  # 相关性项
        ], dim=1)
        geo_feat = self.encoder(invariants)
        x = torch.cat([data.x, geo_feat], dim=1)  # [batch_size * num_nodes, node_dim + 32]

        # 图卷积层
        x = F.relu(self.bn1(self.conv1(x, data.edge_index, data.edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, data.edge_index, data.edge_attr)))
        x = self.conv3(x, data.edge_index, data.edge_attr)  # [batch_size * num_nodes, out_dim]

        # 池化处理
        # global_feat = scatter_mean(x, data.batch, dim=0)  # [batch_size, out_dim]
        unique_batch = torch.unique(data.batch, return_counts=True)
        batch_size = len(unique_batch[0])
        num_nodes = unique_batch[1]
        global_feat = torch.zeros(batch_size, x.size(1), device=x.device)
        start = 0
        for i, num in enumerate(num_nodes):
            global_feat[i] = x[start: start + num].mean(dim=0)
            start += num

        return global_feat


def load_protein_graph(uniprot_id, graph_dir):
    path = Path(graph_dir) / f'{uniprot_id}.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return Data(
        x=data['x'],
        edge_index=data['edge_index'],
        edge_attr=data['edge_attr'],
        pos=data['pos']
    )


if __name__ == '__main__':
    protein_data1 = load_protein_graph('A0A0A1H8I4', 'protein_graphs')
    protein_data2 = load_protein_graph('A0A0B4JD64', 'protein_graphs')
    protein_data3 = load_protein_graph('A0A0C5URS1', 'protein_graphs')
    model = ProteinGNN()
    model.eval()
    batch = Batch.from_data_list([protein_data1, protein_data2, protein_data3])
    with torch.no_grad():
        batch_features = model(batch)
    print(f'批量输入特征形状: {batch_features.shape}')
