import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, accuracy_score, root_mean_squared_error
from scipy.stats import pearsonr
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

# ===== 自定义模块 =====
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
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )

    def forward(self, g2s_features, esm_features):
        g2s_output = self.G2S_feature_learning_model(g2s_features)
        esm_output = self.ESM_feature_learning_model(esm_features)

        g2s_weighted = g2s_output * self.G2S_weight
        esm_weighted = esm_output * self.ESM_weight

        x = g2s_weighted + esm_weighted
        x = x.unsqueeze(1)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def predict(ESM_fea, G2S_fea):
    # 五个模型的文件路径列表
    model_paths = ["Linear_model/Fold{}Model.pth".format(foldi) for foldi in range(5)]

    # 初始化一个列表用于存储五个模型的预测结果
    all_predictions = []

    """""
    'H': [],  # α-螺旋
    'B': [],  # β-桥
    'E': [],  # β-折叠
    'G': [],  # 3-10螺旋
    'I': [],  # π-螺旋
    'T': [],  # 转角
    'S': [],  # 弯曲
    '-': []  # 无规则卷曲
    """
    # 循环遍历五个模型
    for foldi, model_path in enumerate(model_paths):
        # 加载数据

        ESM_test = torch.tensor(ESM_fea).float()
        G2S_test = torch.tensor(G2S_fea).float()

        # 创建模型实例
        model = MyCNNNet()

        # 加载保存的模型参数
        model.load_state_dict(torch.load(model_path))
        # 设置模型为评估模式
        model.eval()
        # 使用模型进行预测
        with torch.no_grad():
            y_pred = model(G2S_test, ESM_test)

        # 将预测结果添加到列表中
        all_predictions.append(y_pred)
        # 计算PCC

    # 计算五个模型预测结果的平均值
    average_predictions = torch.mean(torch.stack(all_predictions), dim=0)
    # print(average_predictions)

    return average_predictions

# ===== 主程序入口 =====
def main(args):
    test_data = np.load(args.test_path)
    print(f"Loaded test data shape: {test_data.shape}")

    ESM_test = test_data[:, :2560]
    G2S_test = test_data[:, -39:-1]
    test_label = test_data[:, -1]
    
    ESM_test = torch.tensor(ESM_test).float()
    G2S_test = torch.tensor(G2S_test).float()
    test_label = torch.tensor(test_label).float().unsqueeze(1)

    test_dataset = TensorDataset(G2S_test, ESM_test, test_label)
    test_loader = DataLoader(test_dataset, shuffle=False)

    all_predictions = []

    for foldi in range(5):
        model_path = os.path.join(args.model_dir, f"Fold{foldi}Model.pth")
        model = MyCNNNet()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        with torch.no_grad():
            y_pred = model(G2S_test, ESM_test)

        all_predictions.append(y_pred)

        pcc_ind, _ = pearsonr(test_label.view(-1).numpy(), y_pred.view(-1).numpy())
        y_pred_sign = (y_pred >= 0).int()
        y_true_sign = (test_label >= 0).int()
        acc_ind = accuracy_score(y_true_sign.numpy(), y_pred_sign.numpy())
        rmse_ind = root_mean_squared_error(test_label, y_pred)

        print(f"Model Fold {foldi}: PCC = {pcc_ind:.4f}, ACC = {acc_ind:.4f}, RMSE = {rmse_ind:.4f}")

    # 平均预测
    average_predictions = torch.mean(torch.stack(all_predictions), dim=0)
    pcc, _ = pearsonr(test_label.view(-1).numpy(), average_predictions.view(-1).numpy())
    y_pred_sign = (average_predictions >= 0).int()
    y_true_sign = (test_label >= 0).int()
    acc = accuracy_score(y_true_sign.numpy(), y_pred_sign.numpy())
    rmse = root_mean_squared_error(test_label, average_predictions)

    print(f"\n>>> Final Evaluation (Average over 5 Folds):")
    print(f"PCC:  {pcc:.4f}")
    print(f"ACC:  {acc:.4f}")
    print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MyCNNNet on test dataset")
    parser.add_argument("--test_path", type=str, required=True, help="Path to .npy test data")
    parser.add_argument("--model_dir", type=str, default="./Linear_model", help="Directory containing model Fold0~Fold4")
    args = parser.parse_args()
    main(args)
