import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 加载保存的训练和测试数据
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# 将数据转换为 PyTorch 张量，并确保数据是浮动类型
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # 分类任务使用 long 类型标签
y_test = torch.tensor(y_test, dtype=torch.long)

# 将数据转换为适合 PyTorch 的数据集
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 定义 DataLoader，用于批次训练
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
