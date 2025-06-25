# coding=utf-8
# EnzymeSubstrateDataset

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from pathlib import Path
from molecular_encode import get_molecular_encoder_input_from_files
from torch_geometric.data import Data
from torch_geometric.data import Batch
import json
from rdkit import RDLogger
import torch.nn.functional as F

RDLogger.DisableLog('rdApp.*')


class EnzymeSubstrateDataset(Dataset):
    def __init__(self,
                 data_file,
                 protein_graph_dir='protein_graphs',
                 molecule_vocab_path='data/motif_token.json'):
        self.df = pd.read_excel(data_file)
        self.protein_graph_dir = protein_graph_dir
        with open(molecule_vocab_path) as f:
            self.motif_vocab = json.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uniprot_id = row['Uniprot ID']
        esm2_features = pickle.load(open(f'esm2_feat/{uniprot_id}.pkl', 'rb'))
        enzyme_feature = {
            'esm2': torch.tensor(esm2_features, dtype=torch.float32),
            'structure': self._load_protein_graph(row['Uniprot ID'])
        }
        substrate_feature = get_molecular_encoder_input_from_files(row['molecule ID'])
        label = torch.tensor(row['outcome'], dtype=torch.float32)
        return {
            'enzyme': enzyme_feature,
            'substrate': substrate_feature,
            'label': label
        }

    def _load_protein_graph(self, uniprot_id):
        path = Path(self.protein_graph_dir) / f"{uniprot_id}.pkl"
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data


def collate_fn(batch):
    enzyme_batch = {
        'esm2': torch.stack([item['enzyme']['esm2'] for item in batch]),
        'structure': Batch.from_data_list([Data(**item['enzyme']['structure']) for item in batch])
    }

    max_atoms = max(item['substrate']['atom']['x'].size(0) for item in batch)
    padded_atom_features = {
        'x': torch.stack([
            F.pad(item['substrate']['atom']['x'],
                  (0, 0, 0, max_atoms - item['substrate']['atom']['x'].size(0)),
                  value=0)
            for item in batch
        ]),  # 形状: [batch_size, max_atoms, feat_dim]
        'adj': torch.stack([
            F.pad(item['substrate']['atom']['adj'],
                  (0, max_atoms - item['substrate']['atom']['adj'].size(0),
                   0, max_atoms - item['substrate']['atom']['adj'].size(1)),
                  value=0)
            for item in batch
        ]),  # 形状: [batch_size, max_atoms, max_atoms]
        'dist': torch.stack([
            F.pad(item['substrate']['atom']['dist'],
                  (0, max_atoms - item['substrate']['atom']['dist'].size(0),
                   0, max_atoms - item['substrate']['atom']['dist'].size(1)),
                  value=float('inf'))
            for item in batch
        ]),  # 形状: [batch_size, max_atoms, max_atoms]
        'mask': torch.stack([
            torch.cat([
                torch.ones(item['substrate']['atom']['x'].size(0)),
                torch.zeros(max_atoms - item['substrate']['atom']['x'].size(0))
            ])
            for item in batch
        ])  # 形状: [batch_size, max_atoms]
    }

    max_motifs = max(item['substrate']['motif']['sum_atoms'].size(0) for item in batch)
    max_atom_map_dim = max(item['substrate']['motif']['atom_map'].size(1) for item in batch)
    padded_motif_features = {
        'ids': torch.stack([
            F.pad(item['substrate']['motif']['ids'],
                  (0, max_motifs - item['substrate']['motif']['ids'].size(0)),
                  value=0)
            for item in batch
        ]),
        'adj': torch.stack([
            F.pad(item['substrate']['motif']['adj'],
                  (0, max_motifs - item['substrate']['motif']['adj'].size(0),
                   0, max_motifs - item['substrate']['motif']['adj'].size(1)),
                  value=0)
            for item in batch
        ]),
        'dist': torch.stack([
            F.pad(item['substrate']['motif']['dist'],
                  (0, max_motifs - item['substrate']['motif']['dist'].size(0),
                   0, max_motifs - item['substrate']['motif']['dist'].size(1)),
                  value=float('inf'))
            for item in batch
        ]),
        'atom_map': torch.stack([
            F.pad(item['substrate']['motif']['atom_map'],
                  (0, max_atom_map_dim - item['substrate']['motif']['atom_map'].size(1),
                   0, max_motifs - item['substrate']['motif']['atom_map'].size(0)),
                  value=0)
            for item in batch
        ]),
        'sum_atoms': torch.stack([
            F.pad(
                item['substrate']['motif']['sum_atoms'].view(-1, 1),
                (0, 0, 0, max_motifs - item['substrate']['motif']['sum_atoms'].size(0)),
                value=1
            )
            for item in batch
        ]),
        'max_atom_map_dim': max_atom_map_dim
    }

    labels = torch.stack([item['label'] for item in batch])

    return {
        'enzyme': enzyme_batch,
        'substrate': {
            'atom': padded_atom_features,
            'motif': padded_motif_features
        },
        'label': labels
    }


if __name__ == "__main__":
    train_dataset = EnzymeSubstrateDataset('data/enzyme_substrate_train_filtereds.xlsx')
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    for batch in train_loader:
        print(batch)
