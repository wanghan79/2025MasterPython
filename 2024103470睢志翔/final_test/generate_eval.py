import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import DEVICE, PDB_OUT_DIR, NPY_OUT_DIR, PLOT_DIR, logger

class Generator:
    def __init__(self, model, model_path: str, seq_template: torch.Tensor):
        self.model = model.to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        self.seq_template = seq_template.to(DEVICE)
        os.makedirs(PDB_OUT_DIR, exist_ok=True)
        os.makedirs(NPY_OUT_DIR, exist_ok=True)

    def sample_latent(self, num_samples: int) -> torch.Tensor:
        return torch.randn(num_samples, LATENT_DIM).to(DEVICE)

    def generate(self, num_samples: int) -> List[torch.Tensor]:
        coords_list = []
        z_samples = self.sample_latent(num_samples)
        with torch.no_grad():
            for idx, z in enumerate(z_samples):
                recon_flat = self.model.decoder(z)
                coords = recon_flat.view(-1, 3)
                coords = coords.cpu()
                coords_list.append(coords)
                npy_path = os.path.join(NPY_OUT_DIR, f"gen_coords_{idx+1}.npy")
                np.save(npy_path, coords.numpy())
                self.write_pdb(coords, os.path.join(PDB_OUT_DIR, f"gen_{idx+1}.pdb"))
        return coords_list

    def write_pdb(self, coords: torch.Tensor, out_path: str):
        L = coords.size(0)
        with open(out_path, 'w') as f:
            atom_idx = 1
            for i in range(L):
                x, y, z = coords[i].tolist()
                f.write(f"ATOM  {atom_idx:5d}  CA  ALA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")
                atom_idx += 1
            f.write("END\n")

class Evaluator:
    def __init__(self, true_coords_list: List[torch.Tensor]):
        self.true_coords_list = true_coords_list

    def compute_rmsd_noalign(self, coords_pred: torch.Tensor, coords_true: torch.Tensor) -> float:
        diff = coords_pred - coords_true
        return torch.sqrt((diff ** 2).sum() / coords_pred.numel()).item()

    def compute_rmsd_kabsch(self, coords_pred: torch.Tensor, coords_true: torch.Tensor) -> float:
        P = coords_pred.numpy()
        Q = coords_true.numpy()
        P_centered = P - P.mean(axis=0)
        Q_centered = Q - Q.mean(axis=0)
        C = np.dot(P_centered.T, Q_centered)
        V, S, Wt = np.linalg.svd(C)
        d = np.sign(np.linalg.det(np.dot(V, Wt)))
        D = np.diag([1.0, 1.0, d])
        U = np.dot(np.dot(V, D), Wt)
        P_rot = np.dot(P_centered, U)
        diff = P_rot - Q_centered
        rmsd = np.sqrt((diff ** 2).sum() / P_rot.size)
        return float(rmsd)

    def evaluate(self, generated_list: List[torch.Tensor], align: bool = True) -> Dict[str, float]:
        rmsds = []
        n = min(len(generated_list), len(self.true_coords_list))
        for i in range(n):
            coords_pred = generated_list[i]
            coords_true = self.true_coords_list[i]
            if align:
                rmsd_val = self.compute_rmsd_kabsch(coords_pred, coords_true)
            else:
                rmsd_val = self.compute_rmsd_noalign(coords_pred, coords_true)
            rmsds.append(rmsd_val)

        mean_rmsd = sum(rmsds) / len(rmsds)
        var_rmsd = sum((x - mean_rmsd) ** 2 for x in rmsds) / len(rmsds)
        rmse_rmsd = math.sqrt(var_rmsd)

        plt.figure(figsize=(8, 6))
        plt.hist(rmsds, bins=20, alpha=0.7)
        plt.xlabel('RMSD (Å)')
        plt.ylabel('Frequency')
        plt.title('RMSD Distribution')
        hist_path = os.path.join(PLOT_DIR, 'rmsd_histogram.png')
        plt.savefig(hist_path)
        plt.close()
        logger.info(f"Saved RMSD histogram to {hist_path}")

        return {'mean_rmsd': mean_rmsd, 'var_rmsd': var_rmsd, 'rmse_rmsd': rmse_rmsd}
