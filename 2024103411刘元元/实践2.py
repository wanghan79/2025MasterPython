import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime


# ====================
# 日志配置
# ====================

def setup_logging():
    """配置日志记录"""
    # 创建日志目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"protein_ssl_{timestamp}.log")

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


# ====================
# 数据集处理模块
# ====================

class ProteinStructureDataset(Dataset):
    def __init__(self, json_files, num_points=1024, augment=True):
        self.json_files = json_files
        self.num_points = num_points
        self.augment = augment
        self.data = self._load_data()  # data只存储点云，不含标签
        self._cached_points = [None] * len(self.data)  # 缓存采样后的点云
        logging.info(f"数据集初始化完成，共加载 {len(self.data)} 个样本")

    def __len__(self):
        """返回数据集中样本的总数"""
        return len(self.data)

    def _load_data(self):
        """加载JSON文件，提取点云"""
        all_points = []
        error_count = 0
        for file in self.json_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                vertices = []
                for triangle in data.values():
                    vertices.extend(triangle)
                vertices = np.array(vertices)
                all_points.append(vertices)
            except Exception as e:
                error_count += 1
                logging.error(f"加载文件 {file} 时出错: {str(e)}")
        assert all(isinstance(p, np.ndarray) for p in all_points), "数据包含非数组元素"
        return all_points

    def _get_random_rotation_matrix(self):
        """生成随机三维旋转矩阵"""
        theta = np.random.uniform(0, 2 * np.pi)  # 绕z轴的旋转角
        phi = np.random.uniform(0, 2 * np.pi)  # 绕y轴的旋转角
        zeta = np.random.uniform(0, 2 * np.pi)  # 绕x轴的旋转角

        # 生成各轴旋转矩阵
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])

        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                       [0, 1, 0],
                       [-np.sin(phi), 0, np.cos(phi)]])

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(zeta), -np.sin(zeta)],
                       [0, np.sin(zeta), np.cos(zeta)]])

        # 组合旋转矩阵 (顺序: Z->Y->X)
        rotation_matrix = np.dot(Rx, np.dot(Ry, Rz))
        return torch.tensor(rotation_matrix, dtype=torch.float32)

    def _rotation_matrix_to_quaternion(self, rotation_matrix):
        """将旋转矩阵转换为四元数"""
        trace = rotation_matrix.trace()
        if trace > 0:
            S = torch.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
        elif (rotation_matrix[0, 0] > rotation_matrix[1, 1]) and (rotation_matrix[0, 0] > rotation_matrix[2, 2]):
            S = torch.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
            qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            qx = 0.25 * S
            qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
            qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            S = torch.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
            qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
            qy = 0.25 * S
            qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
        else:
            S = torch.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
            qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
            qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
            qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
            qz = 0.25 * S
        return torch.stack([qw, qx, qy, qz])

    def __getitem__(self, idx):
        # 采样点云（保持1024个点）
        points = self.data[idx]

        # 统一采样到固定点数
        if len(points) != self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=len(points) < self.num_points)
            points = points[indices]

        # 转换为Tensor
        points = torch.tensor(points, dtype=torch.float32)

        # 数据增强（简化旋转增强）
        if self.augment:
            angle = np.random.uniform(0, 2 * np.pi)
            rot_matrix = torch.tensor([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], dtype=torch.float32)
            points = torch.mm(points, rot_matrix)

            # 生成四元数标签 (绕Z轴旋转)
            rotation_label = torch.tensor([
                np.cos(angle / 2),
                0,
                0,
                np.sin(angle / 2)
            ], dtype=torch.float32)
        else:
            rotation_label = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        return points, rotation_label




# ====================
# 模型架构模块
# ====================

class PointNetLayer(nn.Module):
    """PointNet基础层：1D卷积+批量归一化+ReLU激活"""

    def __init__(self, in_channels, out_channels):
        super(PointNetLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return F.relu(x)


class PointNetEncoder(nn.Module):
    """简化版PointNet++编码器，用于提取点云全局特征"""

    def __init__(self, input_dim=3, global_feat=True):
        """
        参数:
            input_dim: 输入特征维度(默认3D坐标)
            global_feat: 是否返回全局特征
        """
        super(PointNetEncoder, self).__init__()
        self.global_feat = global_feat

        # 特征提取层
        self.layer1 = PointNetLayer(input_dim, 64)
        self.layer2 = PointNetLayer(64, 128)
        self.layer3 = PointNetLayer(128, 256)
        self.layer4 = PointNetLayer(256, 512)

        # 添加全局Dropout层 ▼
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        前向传播过程:
        1. 输入形状: B x N x 3 (批次大小 x 点数 x 坐标维度)
        2. 转置为: B x 3 x N
        3. 通过多层卷积提取特征
        4. 全局最大池化(如果启用)
        """
        # 输入: B x N x 3
        x = x.transpose(2, 1)  # B x 3 x N

        # 特征提取
        x = self.layer1(x)  # B x 64 x N
        x = self.layer2(x)  # B x 128 x N
        x = self.layer3(x)  # B x 256 x N
        x = self.layer4(x)  # B x 512 x N

        # 全局特征池化
        if self.global_feat:
            x = torch.max(x, 2, keepdim=True)[0]  # B x 512 x 1
            x = x.view(-1, 512)  # B x 512
            x = self.dropout(x)

        return x


class ProteinSSLModel(nn.Module):
    """更稳定的模型结构"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B,3,N] -> [B,3,N]
        x = self.encoder(x)     # [B,128,N]
        x = torch.max(x, 2)[0]  # [B,128]
        return F.normalize(self.head(x), dim=1)



class EarlyStopping:
    """早停机制：当验证损失不再改善时停止训练"""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存模型"""
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# ====================
# 训练模块
# ====================

def quaternion_loss(pred, target):
    """简化损失函数：余弦相似度"""
    return 1 - torch.mean(torch.abs(torch.sum(pred * target, dim=1)))


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):  # 移除内部重复定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Dropout(0.2),  # ← 新增正则化
        nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveMaxPool1d(1),
        nn.Flatten(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 4)
    )
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = quaternion_loss  # 使用四元数损失

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        for points, labels in train_loader:
            points = points.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * points.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(device)
                labels = labels.to(device)
                outputs = model(points)
                val_loss += criterion(outputs, labels).item() * points.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        logging.info(f'Epoch {epoch + 1}/{epochs} | '
                     f'Train Loss: {train_loss:.4f} | '
                     f'Val Loss: {val_loss:.4f}')

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curve.png')

    return model


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for points, labels in loader:
            # 确保转换为Tensor后再移动设备
            points = points.to(device)
            labels = labels.to(device)

            outputs = model(points)
            loss = quaternion_loss(outputs, labels)

            # 计算角度误差（以度为单位）
            angle_error = torch.rad2deg(2 * torch.acos(
                torch.clamp(torch.sum(outputs * labels, dim=1).abs(), 0.9999, 1.0)
            ))
            correct += (angle_error < 10.0).sum().item()  # 放宽到10度误差
            total += len(angle_error)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy




# ====================
# 主函数
# ====================

def main():
    # 设置日志
    log_file = setup_logging()
    logging.info("蛋白质自监督学习训练开始")
    logging.info(f"日志文件保存在: {log_file}")

    # 指定包含JSON文件的文件夹路径
    json_folder = 'C:/Users/Lenovo/Desktop/New folder1'  # 请替换为实际文件夹路径
    logging.info(f"从文件夹加载JSON文件: {json_folder}")

    # 获取文件夹中所有JSON文件的路径
    json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder)
                  if f.endswith('.json')]

    # 检查是否有JSON文件
    if not json_files:
        logging.error(f"在文件夹 '{json_folder}' 中未找到JSON文件！")
        return

    logging.info(f"找到 {len(json_files)} 个JSON文件")
    for file in json_files[:5]:  # 只打印前5个文件路径
        logging.debug(f"- {file}")
    if len(json_files) > 5:
        logging.debug(f"... 和 {len(json_files) - 5} 更多文件")

    # 划分训练集和验证集
    if len(json_files) > 1:
        # 如果有多个文件，进行划分
        train_files, test_files = train_test_split(json_files, test_size=0.2, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)
        logging.info(f"数据集划分: 训练集 {len(train_files)}, 验证集 {len(val_files)}, 测试集 {len(test_files)}")
    else:
        # 如果只有一个文件，使用相同文件创建训练集和验证集
        logging.warning("只有一个JSON文件，将使用相同文件进行训练和验证")
        train_files = json_files
        val_files = json_files

    # 创建数据集和数据加载器
    train_dataset = ProteinStructureDataset(train_files)
    val_dataset = ProteinStructureDataset(val_files, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)

    test_dataset = ProteinStructureDataset(test_files, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, drop_last=True)

    sample_points, sample_label = train_dataset[0]
    logging.info(f"样本点云形状: {sample_points.shape}")
    logging.info(f"样本标签: {sample_label} | 范数: {torch.norm(sample_label, p=2):.4f}")

    # 在模型初始化前定义device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 初始化模型
    model = ProteinSSLModel().to(device)
    logging.info(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 记录模型结构
    logging.info("模型结构:")
    for name, param in model.named_parameters():
        logging.debug(f"{name}: {param.shape}")

    # 训练模型
    logging.info("开始训练...")
    # 训练模型
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=50,
        lr=1e-3
    )
    # 保存模型
    model_path = 'protein_ssl_model.pth'
    torch.save(trained_model.state_dict(), model_path)
    logging.info(f"模型训练完成并保存到 {model_path}")

    # 在训练结束后调用
    test_loss, test_acc = evaluate(trained_model, test_loader, device)
    logging.info(f"测试集结果 - Loss: {test_loss:.4f}, Acc: {test_acc:.2%}")

    # 可视化训练曲线
    if os.path.exists('training_curve.png'):
        logging.info("训练曲线已保存为 training_curve.png")


if __name__ == "__main__":
    main()