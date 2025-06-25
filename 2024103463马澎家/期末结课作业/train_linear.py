import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, accuracy_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt

# 初始化早停法所需的变量
best_rmse = float(100)  # 初始化为一个较大的值
early_stopping_patience = 10  # 在停止训练前等待MSE不再改善的轮次
epochs_without_improvement = 0  # 验证集LOSS没有提升的轮次数量

# 存储训练过程中的指标
train_losses = []
test_losses = []
pcc_train_values = []
pcc_test_values = []
acc_train_values = []
acc_test_values = []

# 设置固定的随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def calculate_metrics(predictions, targets):
    """
    计算PCC和ACC

    参数:
    - predictions: 模型的预测结果
    - targets: 真实标签

    返回:
    - pcc: PCC
    - acc: ACC
    """
    # 计算PCC
    pcc = np.corrcoef(predictions.squeeze(), targets.squeeze())[0, 1]

    # 计算ACC
    predictions_sign = torch.sign(predictions)
    targets_sign = torch.sign(targets)
    correct_predictions = (predictions_sign == targets_sign).sum().item()
    total_predictions = len(targets)
    acc = correct_predictions / total_predictions

    return pcc, acc

foldi = 1

ESM_train = np.load("data/train/fold5/integration/S2648_cv_clean{}_train_updated.npy".format(foldi))
ESM_test = np.load("data/train/fold5/integration/S2648_cv_clean{}_test_updated.npy".format(foldi))
G2S_train = np.load("data/train/fold5/integration/S2648_cv_clean{}_train_updated.npy".format(foldi))
G2S_test = np.load("data/train/fold5/integration/S2648_cv_clean{}_test_updated.npy".format(foldi))

# 加入正反向数据
ESM_train = np.vstack((ESM_train, -ESM_train))
ESM_test = np.vstack((ESM_test, -ESM_test))
# 加入正反向数据
G2S_train = np.vstack((G2S_train, -G2S_train))
G2S_test = np.vstack((G2S_test, -G2S_test))

train_label = ESM_train[:, -1]
test_label = ESM_test[:, -1]
ESM_train = ESM_train[:, :-1]
ESM_test = ESM_test[:, :-1]
G2S_train = G2S_train[:, :-1]
G2S_test = G2S_test[:, :-1]

# 转换为PyTorch张量
ESM_train, train_label = torch.tensor(ESM_train).float(), torch.tensor(train_label).float().unsqueeze(1)
G2S_train = torch.tensor(G2S_train).float()

# 转换为PyTorch张量
ESM_test, test_label = torch.tensor(ESM_test).float(), torch.tensor(test_label).float().unsqueeze(1)
G2S_test = torch.tensor(G2S_test).float()

# 创建 Data_dataset 和 DataLoader
train_dataset = TensorDataset(G2S_train, ESM_train, train_label)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = TensorDataset(G2S_test, ESM_test, test_label)
test_loader = DataLoader(test_dataset, shuffle=False)


class HSwishActivation(nn.Module):
    def forward(self, x):
        return x * F.hardtanh((x + 3) / 6, 0, 1)


class G2S_FeatureLearningModel(nn.Module):
    def __init__(self, input_size):
        super(G2S_FeatureLearningModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.hswish = HSwishActivation()
        self.fc2 = nn.Linear(64, 128)
    def forward(self, x):
        x = self.fc1(x)
        x = self.hswish(x)
        x = self.fc2(x)
        return x


class ESM_FeatureLearningModel(nn.Module):
    def __init__(self, input_size):
        super(ESM_FeatureLearningModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc3 = nn.Linear(1024, 128)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        return x

class MyCNNNet(nn.Module):
    def __init__(self):
        super(MyCNNNet, self).__init__()
        self.G2S_feature_learning_model = G2S_FeatureLearningModel(38)
        self.ESM_feature_learning_model = ESM_FeatureLearningModel(2560)

        # 定义两个128维的可学习权重向量
        self.G2S_weight = nn.Parameter(torch.randn(128))
        self.ESM_weight = nn.Parameter(torch.randn(128))

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )
        final_length = 128
        self.fc_layers = nn.Sequential(
            nn.Linear(final_length * 16, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )

    def forward(self, g2s_features, esm_features):
        g2s_output = self.G2S_feature_learning_model(g2s_features)
        esm_output = self.ESM_feature_learning_model(esm_features)

        # 使用按元素相乘的方式进行加权
        g2s_weighted = g2s_output * self.G2S_weight
        esm_weighted = esm_output * self.ESM_weight

        # 将加权后的特征相加
        x = g2s_weighted + esm_weighted
        x = x.unsqueeze(1)  # 为了匹配CNN层的输入维度需求
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# 初始化模型、损失函数和优化器
model = MyCNNNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

# 将模型移至GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型转移到CPU
# device = torch.device("cpu")
model.to(device)

# 训练模型
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    predictions_train = []
    targets_train = []
    for g2s_features, esm_features, labels in train_loader:
        # inputs, labels = inputs.to(device), labels.to(device)
        g2s_features, esm_features, labels = g2s_features.to(device), esm_features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(g2s_features, esm_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * g2s_features.size(0)  # 这里乘的是小批量中的数目，通过循环得到了总的损失
        predictions_train.extend(outputs.cpu().detach().numpy())
        targets_train.extend(labels.cpu().detach().numpy())

    # 在验证集上评估模型
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        predictions_test = []
        targets_test = []
        for g2s_features, esm_features, labels in test_loader:
            g2s_features, esm_features, labels = g2s_features.to(device), esm_features.to(device), labels.to(device)
            outputs = model(g2s_features, esm_features)
            test_loss += criterion(outputs, labels).item() * g2s_features.size(0)  # 这里乘的是小批量中的数目，通过循环得到了总的损失
            predictions_test.extend(outputs.cpu().detach().numpy())
            targets_test.extend(labels.cpu().detach().numpy())
        # 计算平均损失
        train_loss /= len(train_loader.sampler)  # 这里除以总的样本个数得到了平均的损失
        test_loss /= len(test_loader.sampler)
        # 计算RMSE
        train_rmse = np.sqrt(train_loss)
        test_rmse = np.sqrt(test_loss)

        # 计算PCC和ACC
        predictions_train = torch.tensor(np.array(predictions_train)).float().unsqueeze(1)
        targets_train = torch.tensor(np.array(targets_train)).float().unsqueeze(1)

        predictions_test = torch.tensor(np.array(predictions_test)).float().unsqueeze(1)
        targets_test = torch.tensor(np.array(targets_test)).float().unsqueeze(1)

        pcc_train, acc_train = calculate_metrics(predictions_train, targets_train)
        pcc_test, acc_test = calculate_metrics(predictions_test, targets_test)

        # 存储指标值
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        pcc_train_values.append(pcc_train)
        pcc_test_values.append(pcc_test)
        acc_train_values.append(acc_train)
        acc_test_values.append(acc_test)
        # torch.save(model.state_dict(), "test_model/Fold{}Model.pth".format(foldi))
        # print("保存了模型")

        # 检查是否需要早停
        print("best_rmse is {}".format(best_rmse))
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            epochs_without_improvement = 0
            # 保存性能最佳的模型
            torch.save(model.state_dict(), "test_model/Fold{}Model.pth".format(foldi))
            print("保存了最优模型")
        else:
            epochs_without_improvement += 1
            print("epochs_without_improvement = {}".format(epochs_without_improvement))

        # 如果验证集MSE在指定的轮次内没有改善，停止训练
        if epochs_without_improvement >= early_stopping_patience:
            print(f"在经过{epochs_without_improvement}轮没有改善后进行早停法。")
            break

    # 打印或输出其他训练信息
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    print(f'Train rmse: {train_rmse:.4f}, Test rmse: {test_rmse:.4f}')
    print(f'PCC - Train: {pcc_train:.4f}, Test: {pcc_test:.4f}')
    print(f'ACC - Train: {acc_train:.4f}, Test: {acc_test:.4f}')


# 绘制 LOSS 曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('LOSS Curve')
plt.show()

# 绘制 PCC 曲线
plt.figure(figsize=(10, 5))
plt.plot(pcc_train_values, label='PCC - Train')
plt.plot(pcc_test_values, label='PCC - Test')
plt.xlabel('Epoch')
plt.ylabel('PCC')
plt.legend()
plt.title('PCC Curve')
plt.show()

# 绘制 ACC 曲线
plt.figure(figsize=(10, 5))
plt.plot(acc_train_values, label='ACC - Train')
plt.plot(acc_test_values, label='ACC - Test')
plt.xlabel('Epoch')
plt.ylabel('ACC')
plt.legend()
plt.title('ACC Curve')
plt.show()
