import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, X_data, y_data, transform=None):
        self.X_data = X_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        image = self.X_data[idx]
        label = self.y_data[idx]

        # 如果有数据增强（transform），应用它
        if self.transform:
            image = self.transform(image)

        # 转换为Tensor并返回
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)


# 数据增强
def get_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转图像，旋转角度最大为10度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机调整亮度、对比度、饱和度、色调
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),  # 随机仿射变换（平移、旋转等）
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化处理
    ])


# 加载数据并应用增强
def load_data(batch_size=32):
    # 加载数据集
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    # 数据归一化到 [0, 1] 范围
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # 应用数据增强
    train_transform = get_transforms()

    # 数据集对象
    train_dataset = CustomDataset(X_train, y_train, transform=train_transform)
    test_dataset = CustomDataset(X_test, y_test)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
