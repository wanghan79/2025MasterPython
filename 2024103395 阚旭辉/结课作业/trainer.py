import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.gnn_model import GNNEncoder
from models.ged_predictor import GEDPredictor
from utils.data_loader import load_graphs_from_json, load_graph_pairs
from utils.matching_utils import compute_match_score
import dgl

class GedTrainer:
    def __init__(self, dataset_path, input_dim=10, hidden_dim=32, out_dim=16, epochs=50, batch_size=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graphs = load_graphs_from_json(dataset_path)
        self.pairs = load_graph_pairs(self.graphs)
        self.encoder = GNNEncoder(input_dim, hidden_dim, out_dim).to(self.device)
        self.predictor = GEDPredictor(out_dim).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.predictor.parameters()), lr=1e-3)
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        self.encoder.train()
        self.predictor.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for i in range(0, len(self.pairs), self.batch_size):
                batch = self.pairs[i:i+self.batch_size]
                loss = 0.0
                for g1, g2, label in batch:
                    g1, g2, label = g1.to(self.device), g2.to(self.device), label.to(self.device)
                    emb1 = self.encoder(g1, g1.ndata['feat'])
                    emb2 = self.encoder(g2, g2.ndata['feat'])
                    pred = self.predictor(emb1, emb2)
                    loss += F.mse_loss(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

    def evaluate(self):
        self.encoder.eval()
        self.predictor.eval()
        total_loss = 0.0
        with torch.no_grad():
            for g1, g2, label in self.pairs:
                g1, g2, label = g1.to(self.device), g2.to(self.device), label.to(self.device)
                emb1 = self.encoder(g1, g1.ndata['feat'])
                emb2 = self.encoder(g2, g2.ndata['feat'])
                pred = self.predictor(emb1, emb2)
                total_loss += F.mse_loss(pred, label).item()
        print(f"Evaluation Loss: {total_loss / len(self.pairs):.4f}")