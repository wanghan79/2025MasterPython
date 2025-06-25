# coding=utf-8
# molecular_encode

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
import torch
from get_mol import get_mol
import json
from get_motif_token import clean_brics_smiles
import pandas as pd
import pickle


def one_of_k_encoding(x, allowable_set):
    """对x进行one-hot编码，使用给定的允许集合"""
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """对x进行one-hot编码，未知类别使用最后一个元素"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def get_atom_features(atom, use_chirality=True, explicit_H=True):
    """原子特征提取函数"""
    symbol = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
         'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Sn', 'Ag', 'Pd',
         'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Mn', 'Other']
    )
    # 度特征（6维）
    degree = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    # 形式电荷和自由基电子（2维）
    charge_radical = [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
    # 杂化类型（6维）
    hybridization = one_of_k_encoding_unk(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'Other'
        ]
    )
    # 芳香性（1维）
    is_aromatic = [atom.GetIsAromatic()]
    # 氢原子数（5维）
    num_h = []
    if not explicit_H:
        num_h = one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    # 隐式价（6维）
    implicit_valence = one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
    # 手性特征（3维）
    chirality = [False, False, False]
    if use_chirality:
        try:
            chirality = one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            chirality = [False, False] + [atom.HasProp('_ChiralityPossible')]
    # 组合所有特征
    features = (
            symbol + degree + charge_radical + hybridization +
            is_aromatic + num_h + implicit_valence + chirality
    )
    return np.array(features, dtype=np.float32)


def get_bond_features(bond):
    """提取10维键特征"""
    # 键类型（4维）
    bond_type = [
        bond.GetBondType() == Chem.rdchem.BondType.SINGLE,
        bond.GetBondType() == Chem.rdchem.BondType.DOUBLE,
        bond.GetBondType() == Chem.rdchem.BondType.TRIPLE,
        bond.GetBondType() == Chem.rdchem.BondType.AROMATIC
    ]
    # 共轭（1维）
    is_conjugated = [bond.GetIsConjugated()]
    # 环成员（1维）
    is_in_ring = [bond.IsInRing()]
    # 立体化学（4维）
    stereo = one_of_k_encoding_unk(
        str(bond.GetStereo()),
        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
    )
    # 组合所有特征
    features = bond_type + is_conjugated + is_in_ring + stereo
    return np.array(features, dtype=np.float32)


def get_mol_features(mol):
    """获取分子中所有原子的特征"""
    # 获取原子特征
    atom_features = []
    for atom in mol.GetAtoms():
        atom_feat = get_atom_features(atom)
        atom_features.append(atom_feat)
    # 获取键特征并合并到原子特征中
    bond_features = {}
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)
        # 将键特征添加到相连的原子特征中
        if i not in bond_features:
            bond_features[i] = []
        if j not in bond_features:
            bond_features[j] = []
        bond_features[i].append(bond_feat)
        bond_features[j].append(bond_feat)
    # 合并原子和键特征（61维 = 51原子 + 10键）
    full_features = []
    for i, atom_feat in enumerate(atom_features):
        if i in bond_features and bond_features[i]:
            # 对相连键特征取平均
            bond_feat = np.mean(bond_features[i], axis=0).tolist()
        else:
            bond_feat = [0] * 10
        full_feat = atom_feat.tolist() + bond_feat
        full_features.append(full_feat)
    return full_features


def get_motif_decomposition(mol):
    """使用BRICS算法将分子分解为基序(motifs)并添加GLOBAL节点"""
    frag_smiles = BRICS.BRICSDecompose(mol)
    fragments = [Chem.MolFromSmiles(smi) for smi in frag_smiles]
    num_atoms = mol.GetNumAtoms()
    num_motifs = len(fragments)
    # 原子到基序的映射矩阵 [num_motifs+1, num_atoms]
    atom_to_motif = np.zeros((num_motifs + 1, num_atoms), dtype=np.float32)
    # GLOBAL节点连接到所有原子
    atom_to_motif[0, :] = 1
    for i, frag in enumerate(fragments):
        matches = mol.GetSubstructMatches(frag)  # 此处需要修改
        if not matches:
            continue
        matched_indices = matches[0]
        for idx in matched_indices:
            atom_to_motif[i + 1, idx] = 1
    sum_atoms = atom_to_motif.sum(axis=1, keepdims=True)
    return fragments, atom_to_motif, sum_atoms


def get_motif_features(fragments):
    """获取基序的特征表示并添加GLOBAL节点ID"""
    vocab = json.load(open('data/motif_token.json', 'r'))
    motif_ids = [0]
    for frag in fragments:
        smiles = clean_brics_smiles(Chem.MolToSmiles(frag))
        if smiles not in vocab:
            smiles = '[UNK]'
        motif_ids.append(vocab[smiles])
    return np.array(motif_ids, dtype=np.float32), vocab


def build_motif_graph(fragments, atom_to_motif):
    """构建基序图（邻接矩阵和距离矩阵）并添加GLOBAL节点连接"""
    num_motifs = len(fragments) + 1  # 包含GLOBAL节点
    adj_matrix = np.zeros((num_motifs, num_motifs), dtype=np.float32)
    dist_matrix = np.zeros((num_motifs, num_motifs), dtype=np.float32)
    # 构建基序间连接关系
    for i in range(num_motifs):
        for j in range(i, num_motifs):
            if i == 0 or j == 0:
                continue
            shared_atoms = np.sum(atom_to_motif[i] * atom_to_motif[j])
            if shared_atoms > 0:
                adj_matrix[i, j] = adj_matrix[j, i] = 1
                dist_matrix[i, j] = dist_matrix[j, i] = 1.0 / (shared_atoms + 1e-6)
    # 添加GLOBAL节点连接（连接到所有基序）
    adj_matrix[0, :] = 1
    adj_matrix[:, 0] = 1
    # 设置GLOBAL节点的距离（设为平均距离）
    if num_motifs > 2:
        mean_dist = np.mean(dist_matrix[1:, 1:])
    else:
        mean_dist = 1.0
    dist_matrix[0, :] = mean_dist
    dist_matrix[:, 0] = mean_dist
    dist_matrix[0, 0] = 0
    return adj_matrix, dist_matrix


def get_molecular_encoder_input(CHEBI):
    """获取分子编码器输入数据"""
    mol = get_mol(CHEBI)
    # 准备3D结构
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=666)
    AllChem.MMFFOptimizeMolecule(mol)
    # 原子级特征
    atom_features = get_mol_features(mol)
    adj_matrix = torch.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))
    dist_matrix = torch.tensor(AllChem.Get3DDistanceMatrix(mol), dtype=torch.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj_matrix[i, j] = adj_matrix[j, i] = 1
    # 基序级特征
    fragments, atom_to_motif, sum_atoms = get_motif_decomposition(mol)
    motif_ids, _ = get_motif_features(fragments)
    motif_adj, motif_dist = build_motif_graph(fragments, atom_to_motif)
    dic = {
        'atom': {
            'x': torch.FloatTensor(atom_features),  # [N, 61]
            'adj': torch.FloatTensor(adj_matrix),  # [N, N]
            'dist': torch.FloatTensor(dist_matrix)  # [N, N]
        },
        'motif': {
            'ids': torch.LongTensor(motif_ids),  # [M+1]
            'adj': torch.FloatTensor(motif_adj),  # [M+1, M+1]
            'dist': torch.FloatTensor(motif_dist),  # [M+1, M+1]
            'atom_map': torch.FloatTensor(atom_to_motif),  # [M+1, N]
            'sum_atoms': torch.FloatTensor(sum_atoms)  # [M+1, 1]
        },
        'num_atoms': mol.GetNumAtoms(),
        'num_motifs': len(motif_ids)
    }
    return dic


def get_molecular_encoder_input_from_files(CHEBI):
    return pickle.load(open(f'sub_input/{CHEBI.replace(":", "_")}.pkl', 'rb'))


if __name__ == "__main__":
    encoder_input = get_molecular_encoder_input('CHEBI:57344')
    print(encoder_input)
    # df1 = pd.read_excel('data/enzyme_substrate_train.xlsx')
    # df2 = pd.read_excel('data/enzyme_substrate_test.xlsx')
    # df = pd.concat([df1, df2])
    # molecules = list(set(df['molecule ID'].tolist()))
    # for molecule in molecules:
    #     try:
    #         molecular_encoder_input = get_molecular_encoder_input(molecule)
    #         print(molecular_encoder_input['num_atoms'], molecular_encoder_input['num_motifs'])
    #         with open(f'sub_input/{molecule.replace(":", "_")}.pkl', 'wb') as f:
    #             pickle.dump(molecular_encoder_input, f)
    #     except:
    #         print('drop')
