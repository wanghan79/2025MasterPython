import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from config import DATA_DIR, VAL_SPLIT, SEED

class ProteinDataset(Dataset):
    def __init__(self, data_dir: str = DATA_DIR, transform: Any = None):
        super(ProteinDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.entries = []  # List[Tuple[str, str]]: (seq_filename, coord_filename)
        list_path = os.path.join(data_dir, 'list.txt')
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Missing list.txt in {data_dir}")
        with open(list_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) != 2:
                    continue
                seq_file, coord_file = tokens
                self.entries.append((seq_file, coord_file))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_file, coord_file = self.entries[idx]
        seq_path = os.path.join(self.data_dir, 'seqs', seq_file)
        coord_path = os.path.join(self.data_dir, 'coords', coord_file)

        seq_feat = self._load_sequence(seq_path)         # Tensor: (L, 20)
        coords = self._load_coords(coord_path)            # Tensor: (L, 3)
        sample = {'seq_feat': seq_feat, 'coords': coords}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def _load_sequence(self, path: str) -> torch.Tensor:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sequence file not found: {path}")
        with open(path, 'r') as f:
            lines = f.readlines()
        seq = ''.join([l.strip() for l in lines if not l.startswith('>')])
        aa2idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        L = len(seq)
        one_hot = torch.zeros(L, 20, dtype=torch.float32)
        for i, aa in enumerate(seq):
            if aa in aa2idx:
                one_hot[i, aa2idx[aa]] = 1.0
        return one_hot

    def _load_coords(self, path: str) -> torch.Tensor:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Coordinate file not found: {path}")
        if path.endswith('.npy'):
            arr = np.load(path)
            coords = torch.tensor(arr, dtype=torch.float32)
        elif path.endswith('.pdb'):
            coords = load_pdb(path)  # 从utils导入
        else:
            raise ValueError("Unsupported coordinate format: must be .npy or .pdb")
        return coords

class ToGraphTransform:
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seq_feat = sample['seq_feat']     # (L, 20)
        coords = sample['coords']         # (L, 3)
        L = seq_feat.size(0)
        dist_matrix = torch.cdist(coords, coords)  # (L, L)
        adj = (dist_matrix < 8.0).float()
        return {'seq_feat': seq_feat, 'coords': coords, 'adj': adj}

def split_dataset(dataset: Dataset, val_split: float = VAL_SPLIT, seed: int = SEED) -> Tuple[Dataset, Dataset]:
    length = len(dataset)
    val_len = int(length * val_split)
    train_len = length - val_len
    return random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))
