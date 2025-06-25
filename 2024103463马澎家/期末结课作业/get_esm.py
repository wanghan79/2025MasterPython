import os
import torch
import esm
from tqdm import tqdm
from Bio import SeqIO
import numpy as np


def get_esm_fea(wild_seq, mut_seq, model, alphabet, device):
    representation_layers = [model.num_layers]

    print(representation_layers)

    wild = alphabet.encode(wild_seq)
    wild = torch.tensor(wild, device=device).unsqueeze(0)  # 添加 batch 维度并移动到正确的设备

    mut = alphabet.encode(mut_seq)
    mut = torch.tensor(mut, device=device).unsqueeze(0)

    with torch.no_grad():
        wild_results = model(wild, repr_layers=representation_layers)
        mut_results = model(mut, repr_layers=representation_layers)

    wild_embeddings = wild_results['representations'][model.num_layers]
    wild_embeddings = wild_embeddings.squeeze(0).cpu()  # 移动数据回 CPU
    wild_embeddings = torch.mean(wild_embeddings, dim=0)
    # print(f"wild_embedding's shape is {wild_embeddings.shape}")

    mut_embeddings = mut_results['representations'][model.num_layers]
    mut_embeddings = mut_embeddings.squeeze(0).cpu()  # 移动数据回 CPU
    mut_embeddings = torch.mean(mut_embeddings, dim=0)
    # print(f"mut_embedding's shape is {mut_embeddings.shape}")

    embeddings = mut_embeddings - wild_embeddings
    # print(f"embedding's shape is {embeddings.shape} ")

    return embeddings
