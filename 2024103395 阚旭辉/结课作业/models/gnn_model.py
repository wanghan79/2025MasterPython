import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class GNNEncoder(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats):
        super(GNNEncoder, self).__init__()
        self.gcn1 = GraphConv(in_feats, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, out_feats)

    def forward(self, g, feat):
        h = F.relu(self.gcn1(g, feat))
        h = self.gcn2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')