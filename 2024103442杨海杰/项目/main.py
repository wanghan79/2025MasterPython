import os
import random
import re
import json
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from model import Graphormer
from parameter import parse_args

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(0)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    
print(device)
import warnings
warnings.filterwarnings("ignore")


# def count_parameters(model, verbose=True):
#     """计算并打印模型的参数统计信息，包括参数类型"""
#     total_params = 0
#     trainable_params = 0
#     param_details = []
#
#     print("\n模型参数详细统计:")
#     print(f"{'参数名称':<50} {'形状':<30} {'参数数量':<15} {'是否可训练'}")
#     print("-" * 110)
#
#     for name, param in model.named_parameters():
#         param_count = param.numel()
#         is_trainable = param.requires_grad
#
#         # 打印参数详情
#         if verbose:
#             print(f"{name:<50} {str(list(param.shape)):<30} {param_count:<15,} {is_trainable}")
#
#         # 收集统计信息
#         total_params += param_count
#         if is_trainable:
#             trainable_params += param_count
#
#         # 提取参数类型（模块名）
#         module_name = name.split('.')[0] if '.' in name else name
#         param_details.append({
#             'name': name,
#             'module': module_name,
#             'shape': list(param.shape),
#             'count': param_count,
#             'trainable': is_trainable
#         })
#
#     # 按模块分组统计
#     module_stats = {}
#     for param in param_details:
#         module = param['module']
#         if module not in module_stats:
#             module_stats[module] = 0
#         module_stats[module] += param['count']
#
#     # 打印模块级统计
#     print("\n按模块划分的参数数量:")
#     for module, count in sorted(module_stats.items(), key=lambda x: -x[1]):
#         print(f"  {module:<30} {count:,}")
#
#     print(f"\n总参数量: {total_params:,}")
#     print(f"可训练参数量: {trainable_params:,}")
#
#     return total_params, trainable_params, param_details
# 设置中文字体支持
# 设置中文字体支持（根据系统实际字体修改）
plt.rcParams["font.family"] = ["SimHei"]  # 仅保留系统存在的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def train(model, optimizer, x, train_pos_edge_index, train_neg_edge_index, dataset_name, i):
    model.train()
    optimizer.zero_grad()
    z = model(x, train_pos_edge_index, train_neg_edge_index, dataset_name, i)
    loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, x, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index, dataset_name, i):
    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, train_neg_edge_index, dataset_name, i)
    return model.test(z, test_pos_edge_index, test_neg_edge_index)

def get_data(model, dataset_name, i):
    file = f'./train/train.txt'
    test_file = f'./test/test.txt'
    train_data = load_data(file)
    test_data = load_data(test_file)
    offset=4
    train_edge_index, train_edge_attr = process_edges(train_data) #得到边索引和边属性
    train_pos_edge_index = train_edge_index[:, train_edge_attr >=offset ]
    train_neg_edge_index = train_edge_index[:, train_edge_attr <offset]

    test_edge_index, test_edge_attr = process_edges(test_data)
    test_pos_edge_index = test_edge_index[:, test_edge_attr >=offset]
    test_neg_edge_index = test_edge_index[:, test_edge_attr <offset]

    train_pos_edge_index = train_pos_edge_index.to(device)
    train_neg_edge_index = train_neg_edge_index.to(device)
    test_pos_edge_index = test_pos_edge_index.to(device)
    test_neg_edge_index = test_neg_edge_index.to(device)

    pos_edge_index=torch.cat((train_pos_edge_index,test_pos_edge_index),dim=1)
    neg_edge_index=torch.cat((train_neg_edge_index,test_neg_edge_index),dim=1)

    max_node_value = max(torch.max(train_edge_index).item(), torch.max(test_edge_index).item())


    x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index, max_node_value + 1)


    return train_pos_edge_index, test_pos_edge_index, train_neg_edge_index, test_neg_edge_index, x


def plot_training_curve(history, args, dataset_name, seed):
    """绘制训练过程中的损失、AUC、F1和ACC曲线"""
    epochs = range(1, len(history['loss']) + 1)

    # 创建一个2x2的图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f'{dataset_name} 训练曲线 (层数={args.num_layers}, 输出维度={args.output_dim}, 最大度={args.max_degree}, 种子={seed})',
        fontsize=16)

    # 绘制损失曲线
    axes[0, 0].plot(epochs, history['loss'], 'b-', label='训练损失')
    axes[0, 0].set_title('训练损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # 绘制AUC曲线
    axes[0, 1].plot(epochs, history['auc'], 'g-', label='测试AUC')
    axes[0, 1].set_title('测试AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC值')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # 绘制F1曲线
    axes[1, 0].plot(epochs, history['f1'], 'r-', label='测试F1')
    axes[1, 0].set_title('测试F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1值')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # 绘制ACC曲线
    axes[1, 1].plot(epochs, history['acc'], 'm-', label='测试ACC')
    axes[1, 1].set_title('测试ACC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # 保存图表
    os.makedirs('./training_curves', exist_ok=True)
    filename = f"./training_curves/{dataset_name}_{args.num_layers}_{args.output_dim}_{args.max_degree}_{seed}.png"
    plt.savefig(filename)
    print(f"训练曲线已保存至: {filename}")

    plt.close()

def Search(args, dataset_name, seed_list):
    aucs = []
    accs = []
    for i in range(1):
        torch.manual_seed(seed_list[i])
        model = Graphormer(args).to(device)
        # 计算并打印模型参数量
        # total_params, trainable_params, param_details = count_parameters(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

        train_pos_edge_index, test_pos_edge_index, train_neg_edge_index, test_neg_edge_index,  x = get_data(model, dataset_name, i)

        total_time_start = time.time()
        # 用于记录训练历史
        history = {'loss': [], 'auc': [], 'f1': [], 'acc': []}

        for epoch in range(2000):
            loss = train(model, optimizer, x, train_pos_edge_index, train_neg_edge_index, dataset_name, i)
            auc, f1, acc = test(model, x, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index, dataset_name, i)
            # 记录训练数据
            history['loss'].append(loss)
            history['auc'].append(auc)
            history['f1'].append(f1)
            history['acc'].append(acc)

            # 每100个epoch打印一次
            if (epoch + 1) % 100 == 0:
                print(f'Epoch: {epoch + 1}, Loss: {loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, ACC: {acc:.4f}')

            # 绘制训练曲线
        plot_training_curve(history, args, dataset_name, seed_list[i])

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        print(f'Total Time: {total_time:.2f}s, Loss: {loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f},ACC:{acc:.4f}')
        aucs.append(auc)
        accs.append(acc)
        filename = f"./ablation/{args.num_layers}-{args.output_dim}-{args.max_degree}-{i}.json"
        params = {
        'num_layer': args.num_layers,
        'output_dim': args.output_dim,
        'max_degree': args.max_degree,
        'numbers':i,
        'auc':auc,
        'acc':acc
        }
        save_params(params, filename)
    mean_auc = np.mean(aucs)
    mean_acc = np.mean(accs)
    std_auc = np.std(aucs)
    std_acc = np.std(accs)
    # 保存最终性能统计
    stats = {
        'dataset': dataset_name,
        'num_layers': args.num_layers,
        'output_dim': args.output_dim,
        'max_degree': args.max_degree,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'total_time': total_time,
        'auc_values': aucs,
        'acc_values': accs
    }

    os.makedirs('./performance_stats', exist_ok=True)
    stats_filename = f"./performance_stats/{dataset_name}_{args.num_layers}_{args.output_dim}_{args.max_degree}.json"
    save_params(stats, stats_filename)
    print(f"性能统计数据已保存至: {stats_filename}")

    return mean_auc, mean_acc, std_auc, std_acc,total_time

def load_data(file_path):
    if file_path.endswith('.txt'):
        data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format")
    return data


def process_edges(data):
    # Ensure all values are stripped of leading/trailing whitespace
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Convert the first two columns to int and the third column to float
    edge_index = torch.tensor(data.iloc[:, :2].values.astype(np.int64).T, dtype=torch.long)
    edge_attr = torch.tensor(data.iloc[:, 2].values.astype(np.float32), dtype=torch.float)
    return edge_index, edge_attr

def save_params(params, filename):
    with open(filename, 'w') as f:
        json.dump(params, f)

if __name__ == '__main__':
    args = parse_args()
    seed_list = [1145, 14, 68, 9810, 187]

    # param
    num_layers_list = [1]
    output_dim_list = [128]
    max_degree_list = [12]
    dataset1_name = "amazon-music"
    for num_layers in num_layers_list:
        for output_dim in output_dim_list:
            for max_degree in max_degree_list:
                args.num_layers = num_layers
                args.output_dim = output_dim
                args.max_degree = max_degree
                auc_mean,acc_mean,auc_std,acc_std,total_time = Search(args,dataset1_name, seed_list)
    print(f"{dataset1_name}'s auc_mean: {auc_mean},auc_std:{auc_std}")
    print(f"{dataset1_name}'s acc_mean: {acc_mean},acc_std:{acc_std}")
    print(f"{dataset1_name}'s total_time: {total_time}")

