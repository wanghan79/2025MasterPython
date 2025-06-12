import torch
import torch.nn as nn
from config import HIDDEN_DIM, LATENT_DIM

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, latent_dim: int = LATENT_DIM):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = HIDDEN_DIM, output_dim: int = None):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(z))
        out = self.fc2(h)
        return out

class VAE(nn.Module):
    def __init__(self, seq_length: int):
        super(VAE, self).__init__()
        self.seq_length = seq_length
        input_dim = seq_length * 20
        output_dim = seq_length * 3
        self.encoder = Encoder(input_dim, HIDDEN_DIM, LATENT_DIM)
        self.decoder = Decoder(LATENT_DIM, HIDDEN_DIM, output_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, seq_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_flat = seq_feat.view(-1)
        mu, logvar = self.encoder(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_flat = self.decoder(z)
        coords_pred = recon_flat.view(-1, 3)  # (L, 3)
        return coords_pred, mu, logvar
