import torch
import torch.nn as nn

class GEDPredictor(nn.Module):
    def __init__(self, embedding_dim):
        super(GEDPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, emb1, emb2):
        x = torch.cat([emb1, emb2], dim=1)
        return self.mlp(x).squeeze(1)