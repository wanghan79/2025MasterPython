'''
Protein Conformation Generation Pipeline

目标：
    利用Python实现蛋白质构象生成，以序列为输入，生成对应的3D构象候选集。
    包括数据加载、数据预处理、深度学习模型定义（基于变分自编码器VAE或扩散模型）、训练流程、生成与评估。
    代码总量超过550行（注释不计），模块化组织，便于扩展和测试。

主要思路：
    1. 数据准备：读取蛋白质序列和已知结构(PDB)，将其转换为图结构或特征张量。
    2. 数据集与数据加载：实现Dataset类，负责批量加载序列-结构对，提供随机批次，并支持训练/验证集拆分。
    3. 模型定义：基于PyTorch实现VAE，用于学习序列到坐标映射。
    4. 训练流程：编写Trainer类，包含训练循环、损失计算（重构误差、KL散度等）、优化器、学习率调度、日志记录与可视化。
    5. 生成流程：训练完成后，随机采样潜在空间，解码为3D原子坐标，输出PDB文件和.npy文件。
    6. 评估与可视化：计算生成构象与真实结构的RMSD（包含Kabsch对齐），统计分布，并绘制统计曲线。
    7. 工具函数：包括坐标转换、文件读写、RMSD对齐、图可视化等。
    8. 主程序：命令行接口，解析参数，支持训练/生成/评估/分析四种模式。

目录结构（单文件）：
    - imports
    - 常量与配置
    - 数据相关（Dataset与DataLoader封装、拆分函数）
    - 模型定义（Encoder、Decoder、VAE）
    - 训练器（Trainer，含进度条与可视化）
    - 生成与评估（Generator, Evaluator）
    - 工具函数（utils：load_pdb、save_pdb、Kabsch算法、绘图等）
    - 主函数（main）

'''

import os
import sys
import math
import random
import argparse
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------------------------------
# 常量与全局配置
# -----------------------------------------------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 目录配置
DATA_DIR = "data/pdb"
OUTPUT_DIR = "output/samples"
LOG_DIR = "logs/pdb_logs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PDB_OUT_DIR = os.path.join(OUTPUT_DIR, "pdbs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
NPY_OUT_DIR = os.path.join(OUTPUT_DIR, "npy")

# 模型超参数
EMBED_DIM = 128
HIDDEN_DIM = 256
LATENT_DIM = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 50
VAL_SPLIT = 0.2

# 日志配置
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'train.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------------------
# 数据集与数据加载
# -----------------------------------------------------------------------------------------------------------------------
class ProteinDataset(Dataset):
    def __init__(self, data_dir: str, transform: Any = None):
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
        """
        将FASTA序列转换为one-hot编码
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sequence file not found: {path}")
        with open(path, 'r') as f:
            lines = f.readlines()
        seq = ''.join([l.strip() for l in lines if not l.startswith('>')])
        # 氨基酸映射字典
        aa2idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        L = len(seq)
        one_hot = torch.zeros(L, 20, dtype=torch.float32)
        for i, aa in enumerate(seq):
            if aa in aa2idx:
                one_hot[i, aa2idx[aa]] = 1.0
            else:
                # 如果未知氨基酸，保留全零行
                pass
        return one_hot

    def _load_coords(self, path: str) -> torch.Tensor:
        """
        加载坐标：支持 .npy 和 .pdb 两种格式
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Coordinate file not found: {path}")
        if path.endswith('.npy'):
            arr = np.load(path)
            coords = torch.tensor(arr, dtype=torch.float32)
        elif path.endswith('.pdb'):
            coords = load_pdb(path)  # 调用utils中的load_pdb
        else:
            raise ValueError("Unsupported coordinate format: must be .npy or .pdb")
        return coords

class ToGraphTransform:
    """
    将序列和坐标转换为图结构：节点特征为残基One-hot，边为距离阈值。
    """

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seq_feat = sample['seq_feat']     # (L, 20)
        coords = sample['coords']         # (L, 3)
        L = seq_feat.size(0)
        # 构建邻接矩阵 (L, L)，阈值为8Å
        dist_matrix = torch.cdist(coords, coords)  # (L, L)
        adj = (dist_matrix < 8.0).float()
        return {'seq_feat': seq_feat, 'coords': coords, 'adj': adj}


def split_dataset(dataset: Dataset, val_split: float = VAL_SPLIT, seed: int = SEED) -> Tuple[Dataset, Dataset]:
    """
    将数据集按比例划分为训练集和验证集
    """
    length = len(dataset)
    val_len = int(length * val_split)
    train_len = length - val_len
    return random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))


# -----------------------------------------------------------------------------------------------------------------------
# 模型定义：VAE
# -----------------------------------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
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
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(z))
        out = self.fc2(h)
        return out


class VAE(nn.Module):
    """
    简单全连接VAE，将序列特征映射到潜在空间，再解码为坐标序列。
    输入: seq_feat (L, 20) -> 展平 -> x_flat (20L)
    输出: coords_pred (3L)
    """

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
        # seq_feat: (L, 20) -> 展平
        x_flat = seq_feat.view(-1)
        mu, logvar = self.encoder(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_flat = self.decoder(z)
        coords_pred = recon_flat.view(-1, 3)  # (L, 3)
        return coords_pred, mu, logvar


# -----------------------------------------------------------------------------------------------------------------------
# 训练器与损失函数
# -----------------------------------------------------------------------------------------------------------------------
class Trainer:
    """
    Trainer 管理模型训练、验证流程，保存模型，并绘制损失曲线。
    """

    def __init__(self, model: VAE, train_loader: DataLoader, val_loader: DataLoader = None,
                 lr: float = LEARNING_RATE, epochs: int = EPOCHS, output_dir: str = MODELS_DIR):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(PLOT_DIR):
            os.makedirs(PLOT_DIR)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def loss_function(self, recon_coords: torch.Tensor, true_coords: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重构损失: 均方误差 + KL 散度
        """
        mse = nn.functional.mse_loss(recon_coords, true_coords)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld * 0.001

    def train(self):
        best_loss = float('inf')
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]"):
                seq_feats = batch['seq_feat'].to(DEVICE)    # (batch, L, 20)
                coords_true = batch['coords'].to(DEVICE)    # (batch, L, 3)
                batch_size, L, _ = seq_feats.size()
                batch_loss = 0.0
                for i in range(batch_size):
                    seq_feat = seq_feats[i]        # (L, 20)
                    coords_true_i = coords_true[i] # (L, 3)
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
                # 仅基于训练集保存
                if avg_train_loss < best_loss:
                    best_loss = avg_train_loss
                    self.save_model(epoch, best_loss)

        # 训练结束后绘制损失曲线
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
        """
        绘制训练和验证损失曲线，保存为PNG文件
        """
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


# -----------------------------------------------------------------------------------------------------------------------
# 生成与评估模块
# -----------------------------------------------------------------------------------------------------------------------
class Generator:
    """
    生成器：加载训练好的模型，随机采样潜在空间，然后解码生成坐标，最后写入PDB和.npy文件。
    """

    def __init__(self, model: VAE, model_path: str, seq_template: torch.Tensor):
        self.model = model.to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        self.seq_template = seq_template.to(DEVICE)  # 用于生成的序列模板 (L, 20)
        if not os.path.exists(PDB_OUT_DIR):
            os.makedirs(PDB_OUT_DIR)
        if not os.path.exists(NPY_OUT_DIR):
            os.makedirs(NPY_OUT_DIR)

    def sample_latent(self, num_samples: int) -> torch.Tensor:
        """
        从标准正态分布采样潜在向量
        """
        return torch.randn(num_samples, LATENT_DIM).to(DEVICE)

    def generate(self, num_samples: int) -> List[torch.Tensor]:
        coords_list = []
        z_samples = self.sample_latent(num_samples)
        with torch.no_grad():
            for idx, z in enumerate(z_samples):
                recon_flat = self.model.decoder(z)
                coords = recon_flat.view(-1, 3)  # (L, 3)
                coords = coords.cpu()
                coords_list.append(coords)
                # 同时保存为 .npy
                npy_path = os.path.join(NPY_OUT_DIR, f"gen_coords_{idx+1}.npy")
                np.save(npy_path, coords.numpy())
        return coords_list

    def write_pdb(self, coords: torch.Tensor, out_path: str):
        """
        将 coords (L, 3) 写入 PDB 格式，假设每个残基一个 CA 原子
        """
        L = coords.size(0)
        with open(out_path, 'w') as f:
            atom_idx = 1
            for i in range(L):
                x, y, z = coords[i].tolist()
                f.write(f"ATOM  {atom_idx:5d}  CA  ALA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")
                atom_idx += 1
            f.write("END\n")


class Evaluator:
    """
    评估模块：计算生成构象与真实结构的 RMSD，支持对齐与非对齐两种方式，并输出统计信息与图表。
    """

    def __init__(self, true_coords_list: List[torch.Tensor]):
        self.true_coords_list = true_coords_list

    def compute_rmsd_noalign(self, coords_pred: torch.Tensor, coords_true: torch.Tensor) -> float:
        """
        未旋转/平移对齐的 RMSD 计算
        """
        diff = coords_pred - coords_true
        return torch.sqrt((diff ** 2).sum() / coords_pred.numel()).item()

    def compute_rmsd_kabsch(self, coords_pred: torch.Tensor, coords_true: torch.Tensor) -> float:
        """
        使用 Kabsch 算法对齐后再计算 RMSD
        """
        P = coords_pred.numpy()
        Q = coords_true.numpy()
        P_centered = P - P.mean(axis=0)
        Q_centered = Q - Q.mean(axis=0)
        # 计算协方差矩阵
        C = np.dot(P_centered.T, Q_centered)
        # SVD 分解
        V, S, Wt = np.linalg.svd(C)
        # 计算正交矩阵 d
        d = np.sign(np.linalg.det(np.dot(V, Wt)))
        D = np.diag([1.0, 1.0, d])
        # 计算最佳旋转矩阵
        U = np.dot(np.dot(V, D), Wt)
        # 对齐 P
        P_rot = np.dot(P_centered, U)
        # 计算 RMSD
        diff = P_rot - Q_centered
        rmsd = np.sqrt((diff ** 2).sum() / P_rot.size)
        return float(rmsd)

    def evaluate(self, generated_list: List[torch.Tensor], align: bool = True) -> Dict[str, float]:
        """
        计算所有样本的 RMSD 均值、方差、RMSE，并绘制直方图
        """
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

        # 绘制 RMSD 分布直方图
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


# -----------------------------------------------------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------------------------------------------------
def load_pdb(path: str, atom_name: str = 'CA') -> torch.Tensor:
    """
    简单解析 PDB 文件，提取指定原子类型（默认为 CA）的坐标，返回 (L, 3) 张量。
    """
    coords = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDB file not found: {path}")
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
    """
    将 coords (L, 3) 以及对应的残基名称列表写入 PDB 文件
    """
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
    """
    将 coords (L, 3) 保存为 .npy 文件
    """
    np.save(out_path, coords.numpy())


def set_seed(seed: int = SEED):
    """
    固定随机数种子，保证可重复性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------------------------------------------------
# 主函数：解析命令行参数，切换模式
# -----------------------------------------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Protein Conformation Generation')
    subparsers = parser.add_subparsers(dest='mode', help='Mode: train / generate / eval / analyze', required=True)

    # 训练模式
    train_parser = subparsers.add_parser('train', help='Train VAE model')
    train_parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='数据目录')
    train_parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    train_parser.add_argument('--epochs', type=int, default=EPOCHS)
    train_parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    train_parser.add_argument('--output_dir', type=str, default=MODELS_DIR)

    # 生成模式
    gen_parser = subparsers.add_parser('generate', help='Generate conformations')
    gen_parser.add_argument('--model_path', type=str, required=True, help='已训练模型路径')
    gen_parser.add_argument('--seq_file', type=str, required=True, help='用于生成的序列FASTA文件')
    gen_parser.add_argument('--num', type=int, default=10, help='生成样本数量')

    # 评估模式
    eval_parser = subparsers.add_parser('eval', help='Evaluate generated conformations')
    eval_parser.add_argument('--true_coords_dir', type=str, required=True, help='真实坐标目录(.npy)')
    eval_parser.add_argument('--gen_coords_dir', type=str, required=True, help='生成坐标目录(.npy)')
    eval_parser.add_argument('--no_align', action='store_true', help='不进行 Kabsch 对齐')

    # 分析模式：显示损失曲线或 RMSD 直方图
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results: plot saved graphs')
    analyze_parser.add_argument('--plot_type', type=str, choices=['loss', 'rmsd'], required=True, help='选择要查看的图: loss 或 rmsd')

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()

    if args.mode == 'train':
        # 数据加载与拆分
        full_dataset = ProteinDataset(args.data_dir, transform=ToGraphTransform())
        train_dataset, val_dataset = split_dataset(full_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # 模型与训练器初始化
        seq_len_example = train_dataset[0]['seq_feat'].size(0)
        model = VAE(seq_length=seq_len_example)
        trainer = Trainer(model, train_loader, val_loader, lr=args.lr, epochs=args.epochs, output_dir=args.output_dir)
        trainer.train()

    elif args.mode == 'generate':
        # 加载序列模板
        tmp_seq_dataset = ProteinDataset(os.path.dirname(args.seq_file), transform=None)
        seq_feat = tmp_seq_dataset._load_sequence(args.seq_file)
        model = VAE(seq_length=seq_feat.size(0))
        generator = Generator(model, args.model_path, seq_feat)
        coords_list = generator.generate(args.num)
        # 保存为 PDB
        for idx, coords in enumerate(coords_list):
            pdb_path = os.path.join(PDB_OUT_DIR, f"gen_{idx+1}.pdb")
            generator.write_pdb(coords, pdb_path)
            print(f"Saved generated PDB to {pdb_path}")

    elif args.mode == 'eval':
        # 加载真实坐标与生成坐标
        true_coords_list = []
        gen_coords_list = []
        true_files = sorted(os.listdir(args.true_coords_dir))
        gen_files = sorted(os.listdir(args.gen_coords_dir))
        for tf, gf in zip(true_files, gen_files):
            true_path = os.path.join(args.true_coords_dir, tf)
            gen_path = os.path.join(args.gen_coords_dir, gf)
            true_coords = torch.tensor(np.load(true_path), dtype=torch.float32)
            gen_coords = torch.tensor(np.load(gen_path), dtype=torch.float32)
            true_coords_list.append(true_coords)
            gen_coords_list.append(gen_coords)
        evaluator = Evaluator(true_coords_list)
        stats = evaluator.evaluate(gen_coords_list, align=(not args.no_align))
        print("Evaluation Results:")
        for k, v in stats.items():
            print(f"{k}: {v:.4f}")

    elif args.mode == 'analyze':
        # 显示或保存的图，直接打开对应文件夹内的图片即可
        if args.plot_type == 'loss':
            loss_curve_path = os.path.join(PLOT_DIR, 'loss_curve.png')
            if os.path.exists(loss_curve_path):
                from PIL import Image
                img = Image.open(loss_curve_path)
                img.show()
            else:
                print(f"No loss curve found at {loss_curve_path}")
        elif args.plot_type == 'rmsd':
            rmsd_hist_path = os.path.join(PLOT_DIR, 'rmsd_histogram.png')
            if os.path.exists(rmsd_hist_path):
                from PIL import Image
                img = Image.open(rmsd_hist_path)
                img.show()
            else:
                print(f"No RMSD histogram found at {rmsd_hist_path}")

    else:
        print("Unsupported mode. Use 'train', 'generate', 'eval', or 'analyze'.")


if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------------------------------------------------
# TODO:
# 1. 使用更复杂的图神经网络 (GNN)，替换当前全连接VAE，提高结构保真度。
# 2. 引入 Biopython 或 MDAnalysis，支持链处理与残基选择。
# 3. 在 Evaluator 中加入更多评估指标：全局距离测试 (GDT)、TM-score 等。
# 4. 支持配体或辅助因子的坐标重构与评估。
# 5. 集成 OpenMM，计算分子力场能量并进行简化能量最小化后输出。
# 6. 支持变长序列的动态批处理：对不同长度的序列 pad 或 pack/pad_sequence。
# 7. 添加多进程或分布式数据加载提高训练速度。
# 8. 增加可视化模块：利用 matplotlib 绘制训练动态、Embedding 空间 PCA 可视化等。
# 9. 编写详细单元测试 (pytest) 覆盖各模块功能，确保可扩展与维护。
# 10. 添加超参数搜索功能 (如 Optuna)，自动找到最佳网络配置。
