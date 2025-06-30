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
# RobustSum 机制
###########################################################################
class RobustSum(nn.Module):
    """
    在注意力权重 A 与 Value V 上做若干轮迭代，得到更加鲁棒的上下文。
    """
    def __init__(self, L=3, norm="L2", epsilon=1e-2, gamma=4.0, t=1.0, delta=4.0):
        super().__init__()
        self.L = L
        self.norm = norm
        self.epsilon = epsilon
        self.gamma = gamma
        self.t = t
        self.delta = delta

    def forward(self, A, V):
        """
        A: (batch_size, seq_len, seq_len)
        V: (batch_size, seq_len, input_dim)
        """
        M = torch.matmul(A, V)  # (B, seq_len, input_dim)

        for _ in range(self.L):
            dist = torch.cdist(M.detach(), V.detach())  # (B, seq_len, seq_len)

            if self.norm == 'L2':
                w = 1 / (dist + self.epsilon)
            elif self.norm == 'L1':
                w = 1 / (dist + self.epsilon)
            elif self.norm == 'MCP':
                w = 1 / (dist + self.epsilon) - 1 / self.gamma
                w[w < self.epsilon] = self.epsilon
            elif self.norm == 'Huber':
                w = self.delta / (dist + self.epsilon)
                w[w > 1.0] = 1.0

            ww = w * A
            ww_norm = nn.functional.normalize(ww, p=1, dim=-1)
            M = (1.0 - self.t) * M + self.t * torch.matmul(ww_norm, V)

        return M


###########################################################################
# RobustSumSelfAttention: 支持 pool=True/False 参数
###########################################################################
class RobustSumSelfAttention(nn.Module):
    """
    自注意力 + RobustSum：
    - 若 pool=True (默认)，最终返回 (batch_size, input_dim)  [对 seq_len 做平均池化]
    - 若 pool=False，则返回 (batch_size, seq_len, input_dim)  [保留序列形状]
    """
    def __init__(
        self,
        input_dim,
        L=3,
        norm="L2",
        epsilon=1e-2,
        gamma=4.0,
        t=1.0,
        delta=4.0,
        pool=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key   = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.robust_sum = RobustSum(L, norm, epsilon, gamma, t, delta)

        self.pool = pool  # 是否对 seq_len 做最终平均池化

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        B, L, d_model = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scale = (d_model ** 0.5)
        scores = torch.matmul(Q, K.transpose(1, 2)) / scale  # (B, L, L)
        A = torch.softmax(scores, dim=-1)                    # (B, L, L)

        # RobustSum 迭代加权
        context = self.robust_sum(A, V)  # (B, L, d_model)

        if self.pool:
            # 对 seq_len 维度平均 => (B, d_model)
            context = torch.mean(context, dim=1)
        return context


###########################################################################
# Dataset 类
###########################################################################
class SequenceDataset(Dataset):
    def __init__(self, features, targets, seq_length=8):
        super().__init__()
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x_seq = self.features[idx: idx + self.seq_length]
        y_val = self.targets[idx + self.seq_length]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)


###########################################################################
# 双分支 (LSTM + GRU), 保留各自的 Self-Attention, 再用 交叉注意力 进行融合
###########################################################################
class LSTM_GRURegressor(nn.Module):
    """
    - LSTM分支 (双向) => robust self-attn => (B, seq_len, 2h)  [pool=False, 保留序列形状]
    - GRU分支  (双向) => robust self-attn => (B, seq_len, 2h)
    - 交叉注意力: Q=LSTM序列, K=GRU序列, V=GRU序列, + RobustSum => (B, seq_len, 2h)
    - 对 seq_len 做平均 => (B, 2h)
    - BN + Dropout + FC => (B, 1)
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 下分支：LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        # 上分支：GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        # 分支各自的 self-attn (pool=False => 保持序列形状)
        self.att_lstm = RobustSumSelfAttention(
            input_dim=2 * hidden_size,
            pool=False
        )
        self.att_gru = RobustSumSelfAttention(
            input_dim=2 * hidden_size,
            pool=False
        )

        # cross-attn + robust sum
        self.robust_sum = RobustSum(
            L=3, norm="L2", epsilon=1e-2, gamma=4.0, t=1.0, delta=4.0
        )

        # BN + Dropout + FC
        self.bn = nn.BatchNorm1d(2 * hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, x):
        B, L, _ = x.shape

        # ======== LSTM分支 ========
        h0_lstm = torch.zeros(self.num_layers * 2, B, self.hidden_size, device=x.device)
        c0_lstm = torch.zeros(self.num_layers * 2, B, self.hidden_size, device=x.device)
        out_lstm, _ = self.lstm(x, (h0_lstm, c0_lstm))   # (B, L, 2h)

        # 对 LSTM输出做自注意力(robust sum)，pool=False => 仍是 (B, L, 2h)
        out_lstm_seq = self.att_lstm(out_lstm)          # (B, L, 2h)

        # ======== GRU分支 ========
        h0_gru = torch.zeros(self.num_layers * 2, B, self.hidden_size, device=x.device)
        out_gru, _ = self.gru(x, h0_gru)                # (B, L, 2h)

        # 对 GRU输出做自注意力(robust sum)，pool=False => 仍是 (B, L, 2h)
        out_gru_seq = self.att_gru(out_gru)             # (B, L, 2h)

        # ======== 交叉注意力: Q=out_lstm_seq, K=out_gru_seq, V=out_gru_seq ========
        d_model = out_lstm_seq.size(-1)  # 2*hidden_size
        scale = (d_model ** 0.5)
        # (B, L, L)
        attn_scores = torch.matmul(out_lstm_seq, out_gru_seq.transpose(1, 2)) / scale
        A = torch.softmax(attn_scores, dim=-1)  # (B, L, L)

        # robust sum => (B, L, 2h)
        context_seq = self.robust_sum(A, out_gru_seq)

        # ===== 平均池化 => (B, 2h)
        context_mean = torch.mean(context_seq, dim=1)

        # ===== BN + Dropout + FC => (B,1)
        out_cat = self.bn(context_mean)
        out_cat = self.dropout(out_cat)
        out = self.fc(out_cat)
        return out


###########################################################################
# 训练函数
###########################################################################
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-3, device='cpu'):
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

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
            outputs = model(X_batch).view(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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

        scheduler.step(val_avg_loss)
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
    model.eval()
    model.to(device)

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            out = model(X_batch).view(-1)

            out_np = out.cpu().numpy().reshape(-1, 1)
            out_inv = y_scaler.inverse_transform(out_np)

            y_batch_np = y_batch.cpu().numpy().reshape(-1, 1)
            y_batch_inv = y_scaler.inverse_transform(y_batch_np)

            preds_list.append(out_inv)
            targets_list.append(y_batch_inv)

    preds = np.concatenate(preds_list).ravel()
    targets = np.concatenate(targets_list).ravel()

    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))

    station_str = f"站点: {station_name}" if station_name else "未知站点"
    print(f"\n===== {station_str} - Test Results =====")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(targets, label='True')
        plt.plot(preds, label='Pred')
        plt.title(f"Double SelfAttn + CrossAttn(RobustSum) - {station_str}")
        plt.xlabel("Test Sample Index")
        plt.ylabel("能耗值")
        plt.legend()
        fig_name = f"./分析图/{model_type_str}/prediction_{model_type_str}_{station_name}.png" if station_name else f"prediction_{model_type_str}.png"
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.show()

        errors = preds - targets
        # 残差随时间变化
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(errors)), errors)
        plt.title(f"Residual Over Time - {station_str}")
        plt.xlabel("Test Sample Index")
        plt.ylabel("Residual (Pred - True)")
        resid_time_name = f"./分析图/{model_type_str}/error_over_time_{model_type_str}_{station_name}.png" if station_name else f"error_over_time_{model_type_str}.png"
        plt.savefig(resid_time_name, dpi=300, bbox_inches='tight')
        plt.show()

        # 残差直方图
        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=30, alpha=0.7, color='g')
        plt.title(f"Residual Distribution - {station_str}")
        plt.xlabel("Residual (Pred - True)")
        plt.ylabel("Frequency")
        resid_hist_name = f"./分析图/{model_type_str}/error_hist_{model_type_str}_{station_name}.png" if station_name else f"error_hist_{model_type_str}.png"
        plt.savefig(resid_hist_name, dpi=300, bbox_inches='tight')
        plt.show()


###########################################################################
# 主流程
###########################################################################
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # 假设 "simulated_data.csv" 已经存在
    df = pd.read_csv("simulated_data.csv", encoding='utf-8')
    df = df.sort_values(by=['车站编号', 'datetime']).reset_index(drop=True)

    stations = df['车站编号'].unique()

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

        X = df_station[feature_cols].values
        y = df_station[target_col].values

        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_scaled = X_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

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

        n_features = len(feature_cols)
        hidden_size = 64
        num_layers = 1
        dropout = 0.2

        # 保留各分支的自注意力(robust sum) + 最后交叉注意力(robust sum)
        model = LSTM_GRURegressor(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        print(f"[{station}] 正在训练... (Double SelfAttn + CrossAttn(RobustSum))")
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=70,
            learning_rate=1e-3,
            device=device
        )

        # 可视化 Loss
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f"Loss Curve - Station {station} (DoubleSelfAttn+CrossAttn)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        loss_fig_name = f"./分析图/lstm_gru/loss_curve_lstm_gru_{station}.png"
        plt.savefig(loss_fig_name, dpi=300, bbox_inches='tight')
        plt.show()

        # 测试
        evaluate_model(
            model,
            model_type_str="lstm_gru",
            test_loader=test_loader,
            y_scaler=y_scaler,
            device=device,
            plot=True,
            station_name=station
        )

        # 保存权重
        if station == "仁济站-环线":
            safe_station_name = 'renji'
        elif station == "图书馆站-环线":
            safe_station_name = 'tushuguan'
        else:
            safe_station_name = 'shiyoulu'

        model_save_path = f"model_lstm_gru_{safe_station_name}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"模型权重已保存至: {model_save_path}\n\n")


###########################################################################
# 入口
###########################################################################
if __name__ == "__main__":
    main()
