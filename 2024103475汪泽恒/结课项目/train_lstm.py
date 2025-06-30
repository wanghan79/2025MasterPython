import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.font_manager import FontProperties
import re

font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = font.get_name()


###########################################################################
# Dataset 类：将序列数据切片，以供 LSTM 等循环网络使用
###########################################################################
class SequenceDataset(Dataset):
    """
    用于将连续的时序数据 (X) 切分为 [seq_length 个样本, 预测目标] 的形式
    比如 seq_length=8, 则 [X_t, ..., X_{t+7}] -> 预测 y_{t+8}.
    """

    def __init__(self, features, targets, seq_length=8):
        super().__init__()
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x_seq = self.features[idx: idx + self.seq_length]
        y_val = self.targets[idx + self.seq_length]  # 单步预测：下一个时刻的目标

        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)


###########################################################################
# LSTM 回归模型定义
###########################################################################
class LSTMRegressor(nn.Module):
    """
    使用 LSTM 做回归预测：
     - input_size: 每个时刻的输入特征维度
     - hidden_size: LSTM隐藏单元数
     - num_layers: LSTM层数
     - dropout: 每层之间的 dropout 概率
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))   # (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]                     # (batch_size, hidden_size)
        out = self.fc(out)                      # (batch_size, 1)
        return out


###########################################################################
# 训练函数：返回每个 epoch 的 train_loss 和 val_loss
###########################################################################
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-3, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).view(-1)  # (batch_size,)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * X_batch.size(0)

        train_avg_loss = train_loss_total / len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                val_out = model(X_val).view(-1)
                val_loss = criterion(val_out, y_val)
                val_loss_total += val_loss.item() * X_val.size(0)

        val_avg_loss = val_loss_total / len(val_loader.dataset)

        train_losses.append(train_avg_loss)
        val_losses.append(val_avg_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f}")

    print("Training complete.")
    return train_losses, val_losses


###########################################################################
# 测试 & 可视化函数
###########################################################################
def evaluate_model(model, model_type_str, test_loader, y_scaler, device='cpu', plot=True, station_name=None):
    """
    在测试集上预测并计算指标，若 plot=True 则画图对比、并绘制误差图。
    station_name: 可选，若传入则在可视化标题中显示站点名称
    """
    model.eval()
    model.to(device)

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            out = model(X_batch).view(-1)  # shape: (batch_size,)

            # 反标准化回原值
            out_np = out.cpu().numpy().reshape(-1, 1)
            out_inv = y_scaler.inverse_transform(out_np)

            y_batch_np = y_batch.cpu().numpy().reshape(-1, 1)
            y_batch_inv = y_scaler.inverse_transform(y_batch_np)

            preds_list.append(out_inv)
            targets_list.append(y_batch_inv)

    # 拼接完整预测与真实值
    preds = np.concatenate(preds_list).ravel()
    targets = np.concatenate(targets_list).ravel()

    # ========== 计算整体评估指标 ==========
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))

    print(f"\n===== 站点: {station_name if station_name else ''} Test Results =====")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    if plot:
        # ========== 预测 vs. 真实值 ==========
        plt.figure(figsize=(10, 5))
        plt.plot(targets, label='True')
        plt.plot(preds, label='Pred')
        plt.title(f"Single LSTM - {station_name if station_name else ''}")
        plt.xlabel("Test Sample Index")
        plt.ylabel("能耗值")
        plt.legend()
        fig_name = f"./分析图/{model_type_str}/prediction_{model_type_str}_{station_name}.png" if station_name else f"prediction_{model_type_str}.png"
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.show()

        # ========== 残差可视化 ==========
        errors = preds - targets

        # --- 残差随时间变化 ---
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(errors)), errors)
        plt.title(f"Residual Over Time - {station_name if station_name else ''}")
        plt.xlabel("Test Sample Index")
        plt.ylabel("Residual (Pred - True)")
        resid_time_name = f"./分析图/{model_type_str}/error_over_time_{model_type_str}_{station_name}.png" if station_name else f"error_over_time_{model_type_str}.png"
        plt.savefig(resid_time_name, dpi=300, bbox_inches='tight')
        plt.show()

        # --- 残差直方图（分布） ---
        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=30, alpha=0.7, color='g')
        plt.title(f"Residual Distribution - {station_name if station_name else ''}")
        plt.xlabel("Residual (Pred - True)")
        plt.ylabel("Frequency")
        resid_hist_name = f"./分析图/{model_type_str}/error_hist_{model_type_str}_{station_name}.png" if station_name else f"error_hist_{model_type_str}.png"
        plt.savefig(resid_hist_name, dpi=300, bbox_inches='tight')
        plt.show()


###########################################################################
# 主流程：读取数据 -> 分站点训练 & 预测
###########################################################################
def main():

    model_type = "lstm"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # ========== 读取已经合并好并做完特征工程的CSV ==========
    df = pd.read_csv("simulated_data.csv", encoding='utf-8')
    # 确保时间顺序
    df = df.sort_values(by=['车站编号', 'datetime']).reset_index(drop=True)

    # ========== 获取全部站点列表 ==========
    stations = df['车站编号'].unique()

    # ========== 指定使用哪些特征列和目标列 ==========
    feature_cols = [
        '客流量（进出）',
        '环境温度',
        'dayofweek',
        'is_weekend',
        '能耗值_lag1',
        '能耗值_lag2',
        '能耗值_rolling3_mean',
        '能耗值_rolling3_std'
    ]
    target_col = '能耗值'

    for station in stations:
        print("=" * 60)
        print(f"开始处理站点: {station}")
        print("=" * 60)

        df_station = df[df['车站编号'] == station].copy()
        df_station.dropna(subset=feature_cols + [target_col], inplace=True)

        if len(df_station) < 50:
            print(f"站点 {station} 数据量过少，跳过...")
            continue

        # 获取特征矩阵 & 目标
        X = df_station[feature_cols].values  # (N, n_features)
        y = df_station[target_col].values    # (N,)

        # 标准化
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_scaled = X_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # 时序切分：训练/验证/测试
        n = len(X_scaled)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)

        X_train, y_train = X_scaled[:train_end], y_scaled[:train_end]
        X_val, y_val = X_scaled[train_end:val_end], y_scaled[train_end:val_end]
        X_test, y_test = X_scaled[val_end:], y_scaled[val_end:]

        seq_length = 8
        train_dataset = SequenceDataset(X_train, y_train, seq_length)
        val_dataset = SequenceDataset(X_val, y_val, seq_length)
        test_dataset = SequenceDataset(X_test, y_test, seq_length)

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # ======= 构建并训练模型 =======
        n_features = len(feature_cols)
        hidden_size = 64
        num_layers = 1
        dropout = 0.0

        model = LSTMRegressor(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        print(f"[{station}] 正在训练... (模型类型: {model_type})")
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=50,
            learning_rate=1e-3,
            device=device
        )

        # === Loss 曲线 可视化 ===
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f"Loss Curve - Station {station} ({model_type})")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        loss_fig_name = f"./分析图/{model_type}/loss_curve_{model_type}_{station}.png"
        plt.savefig(loss_fig_name, dpi=300, bbox_inches='tight')
        plt.show()

        # === 测试集预测 & 可视化 ===
        evaluate_model(
            model,
            model_type,
            test_loader,
            y_scaler,
            device=device,
            plot=True,
            station_name=station
        )

        if station == "仁济站-环线":
            safe_station_name = 'renji'
        elif station == "图书馆站-环线":
            safe_station_name = 'tushuguan'
        else:
            safe_station_name = 'shiyoulu'

        model_save_path = f"model_{model_type}_{safe_station_name}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"模型权重已保存至: {model_save_path}\n\n")


###########################################################################
# 入口
###########################################################################
if __name__ == "__main__":
    main()
