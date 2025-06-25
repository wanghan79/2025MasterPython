import os
import math
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


# 配置类 - 用于存储模型和训练的配置参数
class Config:
    def __init__(self):
        # 数据配置
        self.data_dir = "./data"
        self.img_size = 256
        self.heatmap_size = 64
        self.num_joints = 17

        # 模型配置
        self.backbone = "resnet50"
        self.use_deconv = True
        self.num_deconv_layers = 3
        self.num_deconv_filters = [256, 256, 256]
        self.num_deconv_kernels = [4, 4, 4]

        # 训练配置
        self.batch_size = 32
        self.num_epochs = 100
        self.lr = 0.001
        self.lr_decay = 0.1
        self.lr_decay_epoch = [70, 90]
        self.weight_decay = 1e-4

        # 小目标处理配置
        self.small_object_threshold = 0.1  # 面积比例阈值
        self.small_object_weight = 2.0  # 小目标损失权重

        # 后处理配置
        self.detection_threshold = 0.1  # 关节点检测阈值
        self.nms_threshold = 0.3  # 非极大值抑制阈值


# 数据增强优化 - 增加小目标关节点的出现频率和多样性
class JointsDataset(Dataset):
    def __init__(self, data_dir: str, is_train: bool = True,
                 augment: bool = True, img_size: int = 256,
                 heatmap_size: int = 64, config=None):
        self.data_dir = data_dir
        self.is_train = is_train
        self.augment = augment
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.config = config or Config()
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """加载数据集"""
        data = []

        # 模拟数据加载
        if self.is_train:
            num_samples = 1000
        else:
            num_samples = 200

        for i in range(num_samples):
            img_path = f"{self.data_dir}/images/img_{i}.jpg"

            # 生成模拟关节点数据 (x, y, visibility)
            joints = np.random.rand(self.config.num_joints, 3) * self.img_size
            joints[:, 2] = np.random.randint(0, 2,
                                             self.config.num_joints)  # 可见性

            # 生成边界框
            x_min = max(0, np.min(joints[:, 0]) - 10)
            y_min = max(0, np.min(joints[:, 1]) - 10)
            x_max = min(self.img_size, np.max(joints[:, 0]) + 10)
            y_max = min(self.img_size, np.max(joints[:, 1]) + 10)

            # 随机生成一些小目标
            if random.random() < 0.3:
                scale_factor = random.uniform(0.1, 0.4)
                center_x = random.uniform(x_min, x_max)
                center_y = random.uniform(y_min, y_max)
                width = (x_max - x_min) * scale_factor
                height = (y_max - y_min) * scale_factor

                x_min = max(0, center_x - width / 2)
                y_min = max(0, center_y - height / 2)
                x_max = min(self.img_size, center_x + width / 2)
                y_max = min(self.img_size, center_y + height / 2)

                # 调整小目标关节点
                for j in range(self.config.num_joints):
                    if joints[j, 2] > 0:
                        joints[j, 0] = x_min + random.uniform(0, 1) * (
                                    x_max - x_min)
                        joints[j, 1] = y_min + random.uniform(0, 1) * (
                                    y_max - y_min)

            data.append({
                'img_path': img_path,
                'joints': joints,
                'bbox': [x_min, y_min, x_max, y_max],  # x1, y1, x2, y2
                'area': (x_max - x_min) * (y_max - y_min)
            })

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        joints = item['joints'].copy()
        bbox = item['bbox']
        area = item['area']

        # 模拟读取图像
        img = np.random.rand(self.img_size, self.img_size, 3) * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)

        # 数据增强
        if self.augment and self.is_train:
            # 随机旋转 (-30° 到 30°)
            rot = np.random.uniform(-30, 30)
            img = TF.rotate(img, rot)
            joints = self._rotate_joints(joints, rot, img.size[0] // 2,
                                         img.size[1] // 2)

            # 随机缩放 (0.8 到 1.2 倍)
            scale = np.random.uniform(0.8, 1.2)
            img = TF.resize(img, (
            int(img.size[1] * scale), int(img.size[0] * scale)))
            joints[:, :2] *= scale
            bbox = [b * scale for b in bbox]

            # 随机水平翻转
            if random.random() < 0.5:
                img = TF.hflip(img)
                joints = self._flip_joints(joints, img.size[0])

            # 亮度/对比度调整
            brightness_factor = np.random.uniform(0.7, 1.3)
            contrast_factor = np.random.uniform(0.7, 1.3)
            img = TF.adjust_brightness(img, brightness_factor)
            img = TF.adjust_contrast(img, contrast_factor)

        # 裁剪到边界框区域并调整为统一大小
        img = TF.crop(img, int(bbox[1]), int(bbox[0]),
                      int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0]))
        img = TF.resize(img, (self.img_size, self.img_size))

        # 调整关节点坐标到裁剪后的图像
        joints[:, 0] -= bbox[0]
        joints[:, 1] -= bbox[1]
        joints[:, 0] *= self.img_size / (bbox[2] - bbox[0])
        joints[:, 1] *= self.img_size / (bbox[3] - bbox[1])

        # 转换为张量
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)

        # 生成热图
        heatmaps = self._generate_heatmaps(joints)

        # 标记小目标关节点
        small_joints_mask = self._get_small_joints_mask(joints, area)

        return {
            'image': img,
            'heatmaps': heatmaps,
            'joints': torch.tensor(joints, dtype=torch.float32),
            'small_joints_mask': torch.tensor(small_joints_mask,
                                              dtype=torch.float32),
            'area': area
        }

    def _rotate_joints(self, joints: np.ndarray, rot: float, cx: float,
                       cy: float) -> np.ndarray:
        """旋转关节点"""
        rot_rad = rot * np.pi / 180
        cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)

        for i in range(joints.shape[0]):
            if joints[i, 2] > 0:  # 如果关节点可见
                x, y = joints[i, 0] - cx, joints[i, 1] - cy
                joints[i, 0] = cx + x * cos_r - y * sin_r
                joints[i, 1] = cy + x * sin_r + y * cos_r
        return joints

    def _flip_joints(self, joints: np.ndarray, width: int) -> np.ndarray:
        """水平翻转关节点"""
        # 左右关节点交换 (COCO格式)
        flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                      [13, 14], [15, 16]]

        joints[:, 0] = width - joints[:, 0] - 1

        for pair in flip_pairs:
            joints[pair[0]], joints[pair[1]] = joints[pair[1]].copy(), joints[
                pair[0]].copy()

        return joints

    def _generate_heatmaps(self, joints: np.ndarray) -> torch.Tensor:
        """生成关节点热图"""
        num_joints = joints.shape[0]
        heatmaps = np.zeros((num_joints, self.heatmap_size, self.heatmap_size),
                            dtype=np.float32)

        sigma = 2.0  # 高斯核大小

        for i in range(num_joints):
            if joints[i, 2] > 0:  # 如果关节点可见
                # 转换到热图尺寸
                x, y = int(joints[i, 0] * self.heatmap_size / self.img_size), \
                    int(joints[i, 1] * self.heatmap_size / self.img_size)

                # 确保在边界内
                if 0 <= x < self.heatmap_size and 0 <= y < self.heatmap_size:
                    # 生成高斯分布
                    heatmap = np.zeros((self.heatmap_size, self.heatmap_size),
                                       dtype=np.float32)
                    for h in range(self.heatmap_size):
                        for w in range(self.heatmap_size):
                            dist_sq = (h - y) * (h - y) + (w - x) * (w - x)
                            exponent = dist_sq / 2.0 / sigma / sigma
                            if exponent > 4.6052:  # exp(-4.6052) = 0.01
                                continue
                            heatmap[h, w] = math.exp(-exponent)

                    # 归一化
                    if np.max(heatmap) > 1e-6:
                        heatmap /= np.max(heatmap)

                    heatmaps[i] = heatmap

        return torch.tensor(heatmaps, dtype=torch.float32)

    def _get_small_joints_mask(self, joints: np.ndarray,
                               area: float) -> np.ndarray:
        """标记小目标关节点"""
        image_area = self.img_size * self.img_size
        small_object_mask = np.zeros(joints.shape[0], dtype=np.float32)

        # 如果区域面积小于阈值，则认为是小目标
        if area / image_area < self.config.small_object_threshold:
            small_object_mask[:] = 1.0

        return small_object_mask


# 模型结构改进 - 加强浅层特征利用，减少信息损失
class DeconvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super(DeconvModule, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PoseEstimationModel(nn.Module):
    def __init__(self, config: Config):
        super(PoseEstimationModel, self).__init__()
        self.config = config

        # 骨干网络 - 使用ResNet50
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50',
                                pretrained=True)

        # 移除最后的全连接层和池化层
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 浅层特征
            resnet.layer2,  # 中层特征
            resnet.layer3,  # 深层特征
            resnet.layer4  # 更深层特征
        )

        # 获取骨干网络的输出通道数
        backbone_out_channels = 2048

        # 反卷积层 - 用于上采样
        if config.use_deconv:
            self.deconv_layers = self._make_deconv_layer(
                config.num_deconv_layers,
                config.num_deconv_filters,
                config.num_deconv_kernels
            )
            final_in_channels = config.num_deconv_filters[-1]
        else:
            self.deconv_layers = None
            final_in_channels = backbone_out_channels

        # 最终预测层
        self.final_layer = nn.Conv2d(
            final_in_channels, config.num_joints,
            kernel_size=1, stride=1, padding=0
        )

        # 浅层特征融合
        self.shallow_fusion = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 中层特征融合
        self.mid_fusion = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def _make_deconv_layer(self, num_layers: int, num_filters: List[int],
                           num_kernels: List[int]) -> nn.Sequential:
        """创建反卷积层"""
        layers = []
        in_channels = 2048

        for i in range(num_layers):
            kernel = num_kernels[i]
            padding = 1 if kernel == 4 else 0
            output_padding = 0 if kernel == 4 else 1

            layers.append(
                DeconvModule(in_channels, num_filters[i], kernel_size=kernel,
                             stride=2, padding=padding, bias=False)
            )
            in_channels = num_filters[i]

        return nn.Sequential(*layers)

    def forward(self, x):
        # 提取特征
        x1 = self.backbone[0:5](x)  # 浅层特征 (256 channels)
        x2 = self.backbone[5](x1)  # 中层特征 (512 channels)
        x3 = self.backbone[6](x2)  # 深层特征 (1024 channels)
        x4 = self.backbone[7](x3)  # 更深层特征 (2048 channels)

        # 特征融合
        x_shallow = F.interpolate(self.shallow_fusion(x1), size=x4.size()[2:],
                                  mode='bilinear', align_corners=False)
        x_mid = F.interpolate(self.mid_fusion(x2), size=x4.size()[2:],
                              mode='bilinear', align_corners=False)

        # 融合多尺度特征
        x = x4 + 0.5 * x_shallow + 0.3 * x_mid

        # 反卷积上采样
        if self.config.use_deconv:
            x = self.deconv_layers(x)

        # 最终预测
        heatmaps = self.final_layer(x)

        return heatmaps


# 训练策略调优 - 调整损失函数和超参数，强化对小目标的关注
class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight: bool = True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None,
                small_joints_mask=None, config=None):
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            if self.use_target_weight:
                if target_weight is not None:
                    loss += 0.5 * self.criterion(
                        heatmap_pred.mul(target_weight[:, idx]),
                        heatmap_gt.mul(target_weight[:, idx])
                    )
                else:
                    loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        # 小目标增强
        if small_joints_mask is not None and config is not None:
            small_object_loss = 0
            for b in range(batch_size):
                if small_joints_mask[b].sum() > 0:
                    small_joints_idx = \
                    (small_joints_mask[b] > 0).nonzero(as_tuple=True)[0]
                    for idx in small_joints_idx:
                        small_object_loss += 0.5 * self.criterion(
                            heatmaps_pred[idx][b].squeeze(),
                            heatmaps_gt[idx][b].squeeze()
                        )

            if small_joints_mask.sum() > 0:
                loss += config.small_object_weight * small_object_loss / (
                            small_joints_mask.sum() * batch_size)

        return loss / num_joints


# 后处理优化 - 降低小目标的检测阈值，避免漏检
def post_process_heatmaps(heatmaps: torch.Tensor,
                          detection_threshold: float = 0.1,
                          nms_threshold: float = 0.3) -> List[Dict]:
    """处理热图并返回关节点坐标"""
    batch_size, num_joints, height, width = heatmaps.shape
    results = []

    for b in range(batch_size):
        joints = []

        for j in range(num_joints):
            heatmap = heatmaps[b, j].detach().cpu().numpy()

            # 找到最大值点
            max_val = np.max(heatmap)
            max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)

            # 应用阈值
            if max_val >= detection_threshold:
                # 亚像素精确定位
                x, y = max_pos[1], max_pos[0]

                # 检查周围像素
                if x > 0 and x < width - 1 and y > 0 and y < height - 1:
                    dx = 0.5 * (heatmap[y, x + 1] - heatmap[y, x - 1]) / (
                            heatmap[y, x + 1] + heatmap[y, x - 1] + 1e-6
                    )
                    dy = 0.5 * (heatmap[y + 1, x] - heatmap[y - 1, x]) / (
                            heatmap[y + 1, x] + heatmap[y - 1, x] + 1e-6
                    )

                    x += dx
                    y += dy

                # 归一化到 [0, 1]
                x = min(max(x / width, 0.0), 1.0)
                y = min(max(y / height, 0.0), 1.0)

                joints.append({
                    'id': j,
                    'x': x,
                    'y': y,
                    'confidence': max_val
                })

        # 应用非极大值抑制 (简化版)
        joints = _apply_nms(joints, nms_threshold)

        results.append({
            'joints': joints,
            'num_joints': len(joints)
        })

    return results


def _apply_nms(joints: List[Dict], threshold: float) -> List[Dict]:
    """应用非极大值抑制"""
    if not joints:
        return []

    # 按置信度排序
    joints = sorted(joints, key=lambda x: x['confidence'], reverse=True)

    selected = []
    for joint in joints:
        overlap = False

        for sel in selected:
            # 计算欧氏距离
            dist = math.sqrt(
                (joint['x'] - sel['x']) ** 2 +
                (joint['y'] - sel['y']) ** 2
            )

            if dist < threshold:
                overlap = True
                break

        if not overlap:
            selected.append(joint)

    return selected


# 模型训练和评估
class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = PoseEstimationModel(config)
        self.criterion = JointsMSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.lr_decay_epoch,
            gamma=config.lr_decay
        )

        # 创建数据加载器
        train_dataset = JointsDataset(
            config.data_dir,
            is_train=True,
            augment=True,
            img_size=config.img_size,
            heatmap_size=config.heatmap_size,
            config=config
        )

        val_dataset = JointsDataset(
            config.data_dir,
            is_train=False,
            augment=False,
            img_size=config.img_size,
            heatmap_size=config.heatmap_size,
            config=config
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def train(self, device: torch.device):
        """训练模型"""
        self.model.to(device)

        for epoch in range(self.config.num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['image'].to(device)
                heatmaps = batch['heatmaps'].to(device)
                small_joints_mask = batch['small_joints_mask'].to(device)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    heatmaps,
                    small_joints_mask=small_joints_mask,
                    config=self.config
                )

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f'Epoch: {epoch + 1}/{self.config.num_epochs}, '
                          f'Batch: {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item():.6f}')

            # 验证阶段
            val_loss, val_accuracy = self.validate(device)

            # 学习率调整
            self.lr_scheduler.step()

            print(f'Epoch: {epoch + 1}/{self.config.num_epochs}, '
                  f'Train Loss: {train_loss / len(self.train_loader):.6f}, '
                  f'Val Loss: {val_loss:.6f}, '
                  f'Val Accuracy: {val_accuracy:.4f}%')

            # 保存模型
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss
                }, f'model_epoch_{epoch + 1}.pth')

    def validate(self, device: torch.device) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(device)
                heatmaps = batch['heatmaps'].to(device)
                joints = batch['joints'].to(device)
                small_joints_mask = batch['small_joints_mask'].to(device)

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    heatmaps,
                    small_joints_mask=small_joints_mask,
                    config=self.config
                )

                val_loss += loss.item()

                # 计算准确率
                batch_size = images.size(0)
                for b in range(batch_size):
                    pred_joints = post_process_heatmaps(
                        outputs[b:b + 1],
                        detection_threshold=self.config.detection_threshold
                    )[0]['joints']

                    for pred in pred_joints:
                        joint_id = pred['id']
                        if joints[b, joint_id, 2] > 0:  # 如果真实关节点可见
                            # 计算预测点与真实点的距离
                            pred_x, pred_y = pred['x'] * self.config.img_size, \
                                             pred['y'] * self.config.img_size
                            true_x, true_y = joints[b, joint_id, 0], joints[
                                b, joint_id, 1]

                            dist = math.sqrt(
                                (pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)

                            # 如果距离小于阈值，则认为预测正确
                            if dist < 10.0:  # 10像素阈值
                                correct += 1

                            total += 1

        val_loss /= len(self.val_loader)
        accuracy = 100.0 * correct / max(total, 1)

        return val_loss, accuracy


# 主函数
def main():
    config = Config()

    # 设置随机种子，保证结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建训练器
    trainer = Trainer(config)

    # 训练模型
    trainer.train(device)

    # 示例：使用训练好的模型进行预测
    if device.type == "cuda":
        model = PoseEstimationModel(config).cuda()
        checkpoint = torch.load("model_epoch_100.pth")
    else:
        model = PoseEstimationModel(config)
        checkpoint = torch.load("model_epoch_100.pth",
                                map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 模拟一个输入
    dummy_input = torch.randn(1, 3, config.img_size, config.img_size).to(device)

    # 进行预测
    with torch.no_grad():
        output = model(dummy_input)

    # 后处理
    results = post_process_heatmaps(
        output,
        detection_threshold=config.detection_threshold,
        nms_threshold=config.nms_threshold
    )

    print(f"预测结果: {results}")


if __name__ == "__main__":
    main()