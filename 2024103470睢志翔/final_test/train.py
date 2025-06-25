import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from config import DEVICE, MODELS_DIR, PLOT_DIR, logger

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader = None,
                 lr: float = 1e-3, epochs: int = 50, output_dir: str = MODELS_DIR):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(PLOT_DIR, exist_ok=True)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def loss_function(self, recon_coords: torch.Tensor, true_coords: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        mse = nn.functional.mse_loss(recon_coords, true_coords)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld * 0.001

    def train(self):
        best_loss = float('inf')
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]"):
                seq_feats = batch['seq_feat'].to(DEVICE)
                coords_true = batch['coords'].to(DEVICE)
                batch_size, L, _ = seq_feats.size()
                batch_loss = 0.0
                for i in range(batch_size):
                    seq_feat = seq_feats[i]
                    coords_true_i = coords_true[i]
                    coords_pred, mu, logvar = self.model(seq_feat)
                    loss = self.loss_function(coords_pred, coords_true_i, mu, logvar)
                    batch_loss += loss
                batch_loss = batch_loss / batch_size
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                train_loss += batch_loss.item()

            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            logger.info(f"Epoch {epoch}/{self.epochs}, Train Loss: {avg_train_loss:.4f}")
            print(f"Epoch {epoch}/{self.epochs}, Train Loss: {avg_train_loss:.4f}")

            if self.val_loader:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                logger.info(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
                print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model(epoch, best_loss)
            else:
                if avg_train_loss < best_loss:
                    best_loss = avg_train_loss
                    self.save_model(epoch, best_loss)

        self.plot_losses()

    def validate(self) -> float:
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                seq_feats = batch['seq_feat'].to(DEVICE)
                coords_true = batch['coords'].to(DEVICE)
                batch_size, L, _ = seq_feats.size()
                batch_loss = 0.0
                for i in range(batch_size):
                    seq_feat = seq_feats[i]
                    coords_true_i = coords_true[i]
                    coords_pred, mu, logvar = self.model(seq_feat)
                    loss = self.loss_function(coords_pred, coords_true_i, mu, logvar)
                    batch_loss += loss
                val_loss += (batch_loss / batch_size).item()
        return val_loss / len(self.val_loader)

    def save_model(self, epoch: int, loss: float):
        save_path = os.path.join(self.output_dir, f"vae_epoch{epoch}_loss{loss:.4f}.pt")
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Saved model to {save_path}")
        print(f"Saved model to {save_path}")

    def plot_losses(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Train Loss')
        if self.val_loader:
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plot_path = os.path.join(PLOT_DIR, 'loss_curve.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved loss curve to {plot_path}")
