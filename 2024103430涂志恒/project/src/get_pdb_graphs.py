# coding=utf-8
# get_pdb_graphs

import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm


def pdb_to_graph(pdb_path, k=5, cutoff=10.0):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_path)
    # 提取残基坐标和特征
    residues = [res for res in structure.get_residues() if res.id[0] == ' ']
    ca_coords = []
    residue_features = []
    for res in residues:
        ca = res['CA']
        ca_coords.append(ca.coord)
        # 残基特征：氨基酸类型(20D) + 二级结构(3D) + 溶剂可及性(1D)
        residue_features.append(get_residue_feature(res))
    ca_coords = np.array(ca_coords)
    residue_features = np.array(residue_features)  # [N_res, 24]
    # 构建边索引 (KNN + 距离阈值)
    dist_matrix = cdist(ca_coords, ca_coords)
    # KNN连接
    knn_indices = np.argpartition(dist_matrix, k, axis=1)[:, :k + 1]
    knn_mask = np.zeros_like(dist_matrix, dtype=bool)
    for i in range(len(knn_indices)):
        knn_mask[i, knn_indices[i]] = True
    # 距离阈值连接
    dist_mask = (dist_matrix < cutoff) & (dist_matrix > 0)
    # 合并连接
    adj = knn_mask | dist_mask
    # 边索引 [2, E]
    edge_index = np.array(np.where(adj)).astype(int)
    # 边特征
    edge_attrs = []
    for i, j in zip(*edge_index):
        delta = ca_coords[i] - ca_coords[j]
        edge_attrs.append([
            dist_matrix[i, j],  # 欧氏距离
            *delta,  # 相对坐标
            np.linalg.norm(delta)  # 向量模长
        ])
    return Data(
        x=torch.FloatTensor(residue_features),  # 节点特征 [N, 24]
        edge_index=torch.LongTensor(edge_index),  # 边索引 [2, E]
        edge_attr=torch.FloatTensor(edge_attrs),  # 边特征 [E, 5]
        pos=torch.FloatTensor(ca_coords)  # 3D坐标 [N, 3]
    )


def get_residue_feature(residue):
    # 氨基酸类型 (20D one-hot)
    aa_types = 'ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split()
    aa_onehot = np.zeros(20)
    if residue.get_resname() in aa_types:
        aa_onehot[aa_types.index(residue.get_resname())] = 1
    # 二级结构 (3D one-hot)
    ss_types = ['H', 'E', 'C']
    ss_onehot = np.zeros(3)
    if 'SSE' in residue.xtra:
        ss_onehot[ss_types.index(residue.xtra['SSE'])] = 1
    # 溶剂可及性 (1D)
    sasa = residue.xtra.get('SASA', 0.0) if hasattr(residue, 'xtra') else 0.0
    return np.concatenate([aa_onehot, ss_onehot, [sasa]])  # 实际发现后面4个特征全0


def save_protein_graph(pdb_path, output_dir):
    try:
        protein_data = pdb_to_graph(pdb_path)
        uniprot_id = Path(pdb_path).stem
        output_path = Path(output_dir) / f'{uniprot_id}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump({
                'uniprot_id': uniprot_id,
                'x': protein_data.x,  # 节点特征 [N, 24]
                'edge_index': protein_data.edge_index,  # 边索引 [2, E]
                'edge_attr': protein_data.edge_attr,  # 边特征 [E, 5]
                'pos': protein_data.pos  # 3D坐标 [N, 3]
            }, f)
        return True
    except Exception as e:
        print(f'Failed to process {pdb_path}: {str(e)}')
        return False


def preprocess_pdb_batch(pdb_dir, output_dir):
    pdb_files = list(Path(pdb_dir).glob('*.pdb'))
    success = 0
    for pdb_path in tqdm(pdb_files, desc='Processing PDBs'):
        if save_protein_graph(pdb_path, output_dir):
            success += 1
    print(f'\nDone! Success: {success}/{len(pdb_files)}')
    return success


if __name__ == '__main__':
    preprocess_pdb_batch(
        pdb_dir='pdb/',
        output_dir='protein_graphs/'
    )
