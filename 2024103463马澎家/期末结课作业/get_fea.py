import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys
sys.path.append("/root/fssd/SMFFDDG")
from MU3DSPstar import Mu3DSP_dssp_aafs_aap
import esm

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def load_model():
    print("正在加载ESM模型，请稍后.....")
    os.environ['TORCH_HOME'] = 'ESM2_model'  # 可修改为保存模型的路径
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"模型加载完成，使用设备：{device}")
    return model, alphabet, device


def get_esm_fea(wild_seq, mut_seq, model, alphabet, device):
    representation_layers = [model.num_layers]
    wild = torch.tensor(alphabet.encode(wild_seq), device=device).unsqueeze(0)
    mut = torch.tensor(alphabet.encode(mut_seq), device=device).unsqueeze(0)

    with torch.no_grad():
        wild_results = model(wild, repr_layers=representation_layers)
        mut_results = model(mut, repr_layers=representation_layers)

    wild_embeddings = wild_results['representations'][model.num_layers].squeeze(0).cpu()
    mut_embeddings = mut_results['representations'][model.num_layers].squeeze(0).cpu()

    wild_mean = torch.mean(wild_embeddings, dim=0)
    mut_mean = torch.mean(mut_embeddings, dim=0)

    diff_embedding = mut_mean - wild_mean 
    
    return diff_embedding



def process_csv(csv_path, save_path):
    model, alphabet, device = load_model()

    df = pd.read_csv(csv_path)
    print(f"共读取 {len(df)} 条数据")

    all_features = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理中"):
        try:
            wt_seq = row["wt_seq"]
            mut_seq = row["mut_seq"]
            ddg = row["ddg"]
            protein_id = row['id']
            # 解析突变信息
            mut_info = row["mut_info"]
            mutation_res = mut_info[-1]
            position = int(row['pos']) - 1  # 注意这里转为 0-based
            wild_res = mut_info[0]

            # 安全检查，确保 wild_res 和 wt_seq[position] 一致
            """""
            if wt_seq[position] != wild_res:
                print("")
                raise ValueError(f"wild_res 不匹配：蛋白：{protein_id}，position:{position}, mut_info 提供的是 {wild_res}，但 wt_seq[{position}] = {wt_seq[position]}")
            """

            # 生成mutant sequence
            mut_seq_constructed = mut_seq

            # 1. G2S 特征
            g2s_fea = Mu3DSP_dssp_aafs_aap(wild_res, mutation_res, position + 1, wt_seq)
            g2s_fea = np.array(g2s_fea).reshape(1, -1)

            # 2. ESM特征
            esm_fea_tensor = get_esm_fea(wt_seq, mut_seq_constructed, model, alphabet, device)
            esm_fea = esm_fea_tensor.detach().cpu().numpy().reshape(1, -1)

            # 3. 拼接 + 加上标签
            feature = np.concatenate([esm_fea, g2s_fea, np.array([[ddg]])], axis=1)  # shape: (1, dim)
            all_features.append(feature)

        except Exception as e:
            print(f"第 {idx} 行处理失败：{e}")
            esm_dim = model.embed_dim
            g2s_dim = len(Mu3DSP_dssp_aafs_aap("A", "G", 0, "A" * 10))  # 粗略估计
            feature = np.concatenate([np.zeros((1, esm_dim)), np.zeros((1, g2s_dim)), np.zeros((1, 1))], axis=1)
            all_features.append(feature)

    # 拼接所有行 -> shape (N, feature_dim)
    all_features = np.concatenate(all_features, axis=0)
    print(all_features.shape)
    np.save(save_path, all_features)
    print(f"所有特征已保存至 {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="/root/fssd/SMFFDDG/data/test/S669_with_ddg.csv", help="输入CSV文件路径")
    parser.add_argument("--save_path", type=str, default="process_data/s669_full_embeddings_2.npy", help="输出npy文件路径")
    args = parser.parse_args()

    process_csv(args.csv_path, args.save_path)
