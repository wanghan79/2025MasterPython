import os
import numpy as np
import torch
from typing import List

def load_pdb(path: str, atom_name: str = 'CA') -> torch.Tensor:
    coords = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom = line[12:16].strip()
                if atom == atom_name:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
    if not coords:
        raise ValueError(f"No atoms named {atom_name} found in {path}")
    return torch.tensor(coords, dtype=torch.float32)

def save_pdb(coords: torch.Tensor, residue_names: List[str], out_path: str):
    if len(residue_names) != coords.size(0):
        raise ValueError("residue_names length must match number of coordinates")
    with open(out_path, 'w') as f:
        atom_idx = 1
        for i, coord in enumerate(coords):
            x, y, z = coord.tolist()
            res_name = residue_names[i]
            f.write(f"ATOM  {atom_idx:5d}  CA  {res_name} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")
            atom_idx += 1
        f.write("END\n")

def write_npy(coords: torch.Tensor, out_path: str):
    np.save(out_path, coords.numpy())
