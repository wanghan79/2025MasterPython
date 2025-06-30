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
# RobustSum & RobustSumSelfAttention
###########################################################################
class RobustSum(nn.Module):
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
        V: (batch_size, seq_len, d_model)
        """
        M = torch.matmul(A, V)
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
            ww_norm = torch.nn.functional.normalize(ww, p=1, dim=-1)
            M = (1.0 - self.t) * M + self.t * torch.matmul(ww_norm, V)
        return M


class RobustSumSelfAttention(nn.Module):
    """
    自注意力 + RobustSum：
    - 若 pool=False，则返回序列 (B, seq_len, d_model)；若 pool=True，则最后做平均池化 (B, d_model)
    """
    def __init__(self, input_dim, L=3, norm="L2", epsilon=1e-2, gamma=4.0, t=1.0, delta=4.0, pool=True):
        super().__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key   = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.robust_sum = RobustSum(L, norm, epsilon, gamma, t, delta)
        self.pool = pool

    def forward(self, x):
        """
        x: (B, seq_len, d_model)
        """
        B, L, d_model = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scale = (d_model ** 0.5)
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / scale  # (B, L, L)
        A = torch.softmax(attn_scores, dim=-1)  # (B, L, L)

        context = self.robust_sum(A, V)         # (B, L, d_model)

        if self.pool:
            context = torch.mean(context, dim=1)  # (B, d_model)
        return context


###########################################################################
# Dataset 与原训练一致
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
# 旧的单分支 LSTM 模型
###########################################################################
class LSTMRegressor(nn.Module):
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

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


###########################################################################
# 新的双分支 LSTM+GRU，保留分支自注意力 + 交叉注意力 + RobustSum
###########################################################################
class LSTM_GRURegressor(nn.Module):
    """
    双分支模型：LSTM分支 + GRU分支 (双向可选)
    1) 各分支输出 => robust self-attn (pool=False) => (B, L, 2h)
    2) CrossAttn(Q=LSTM_seq, K=GRU_seq, V=GRU_seq) => robust sum => (B, L, 2h)
    3) 对 seq_len 做平均 => (B, 2h)
    4) BN + Dropout + FC => (B,1)
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ============ 分支1: LSTM (双向可改成 True, 这里写False示例) ============
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True  # 改为 True，默认双向
        )

        # ============ 分支2: GRU (双向) ============
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        # 自注意力 (pool=False => 保留序列形状)
        self.att_lstm = RobustSumSelfAttention(
            input_dim=2 * hidden_size, pool=False
        )
        self.att_gru = RobustSumSelfAttention(
            input_dim=2 * hidden_size, pool=False
        )

        # cross-attn => robust sum
        self.cross_robust = RobustSum(
            L=3, norm="L2", epsilon=1e-2, gamma=4.0, t=1.0, delta=4.0
        )

        self.bn = nn.BatchNorm1d(2 * hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, x):
        B, L, _ = x.size()

        # LSTM分支
        h0_lstm = torch.zeros(self.num_layers * 2, B, self.hidden_size, device=x.device)
        c0_lstm = torch.zeros(self.num_layers * 2, B, self.hidden_size, device=x.device)
        out_lstm, _ = self.lstm(x, (h0_lstm, c0_lstm))  # (B, L, 2h)
        # robust self-attn => (B, L, 2h)
        out_lstm_seq = self.att_lstm(out_lstm)

        # GRU分支
        h0_gru = torch.zeros(self.num_layers * 2, B, self.hidden_size, device=x.device)
        out_gru, _ = self.gru(x, h0_gru)  # (B, L, 2h)
        # robust self-attn => (B, L, 2h)
        out_gru_seq = self.att_gru(out_gru)

        # ============ 交叉注意力: Q=out_lstm_seq, K=out_gru_seq, V=out_gru_seq ============
        d_model = out_lstm_seq.size(-1)  # 2h
        scale = (d_model ** 0.5)
        attn_scores = torch.matmul(out_lstm_seq, out_gru_seq.transpose(1, 2)) / scale  # (B, L, L)
        A = torch.softmax(attn_scores, dim=-1)

        context_seq = self.cross_robust(A, out_gru_seq)  # (B, L, 2h)

        # 平均池化 => (B, 2h)
        context_mean = torch.mean(context_seq, dim=1)

        # BN + Dropout + FC => (B,1)
        out_cat = self.bn(context_mean)
        out_cat = self.dropout(out_cat)
        out = self.fc(out_cat)
        return out


###########################################################################
# 推理函数：给定模型 & dataloader，返回反标准化后的预测值和真实值
###########################################################################
def inference(model, test_loader, y_scaler, device='cpu'):
    model.eval()
    model.to(device)

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            out = model(X_batch).view(-1)  # (batch_size,)

            # 反标准化
            out_np = out.cpu().numpy().reshape(-1, 1)
            out_inv = y_scaler.inverse_transform(out_np)

            y_batch_np = y_batch.cpu().numpy().reshape(-1, 1)
            y_batch_inv = y_scaler.inverse_transform(y_batch_np)

            preds_list.append(out_inv)
            targets_list.append(y_batch_inv)

    # 拼接
    preds = np.concatenate(preds_list).ravel()
    targets = np.concatenate(targets_list).ravel()
    return preds, targets


###########################################################################
# 测试主流程：加载数据 -> 构建并载入两个模型权重 -> 对比预测 & 绘图
###########################################################################
def main_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 读取数据
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
        print(f"开始测试站点: {station}")
        print("=" * 60)

        df_station = df[df['车站编号'] == station].copy()
        df_station.dropna(subset=feature_cols + [target_col], inplace=True)

        if len(df_station) < 50:
            print(f"站点 {station} 数据量过少，跳过...")
            continue

        # 特征 & 目标
        X = df_station[feature_cols].values
        y = df_station[target_col].values

        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_scaled = X_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        n = len(X_scaled)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)

        # 只需要测试集
        X_test = X_scaled[val_end:]
        y_test = y_scaled[val_end:]

        seq_length = 8
        test_dataset = SequenceDataset(X_test, y_test, seq_length)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        if station == "仁济站-环线":
            safe_station_name = 'renji'
        elif station == "图书馆站-环线":
            safe_station_name = 'tushuguan'
        else:
            safe_station_name = 'shiyoulu'

        # ========== 构建两个模型并加载权重 ==========

        # LSTM
        lstm_model = LSTMRegressor(
            input_size=len(feature_cols),
            hidden_size=64,
            num_layers=1,
            dropout=0.0
        )
        lstm_ckpt_path = f"model_lstm_{safe_station_name}.pth"
        ok_lstm = True
        try:
            lstm_model.load_state_dict(torch.load(lstm_ckpt_path, map_location=device))
            print(f"已载入 LSTM 模型权重: {lstm_ckpt_path}")
        except:
            print(f"[警告] 未能载入 {lstm_ckpt_path}, 将跳过 LSTM 对比。")
            ok_lstm = False

        # LSTM_GRU
        lstm_gru_model = LSTM_GRURegressor(
            input_size=len(feature_cols),
            hidden_size=64,
            num_layers=1,
            dropout=0.0
        )
        gru_ckpt_path = f"model_lstm_gru_{safe_station_name}.pth"
        ok_gru = True
        try:
            lstm_gru_model.load_state_dict(torch.load(gru_ckpt_path, map_location=device))
            print(f"已载入 LSTM_GRU 模型权重: {gru_ckpt_path}")
        except:
            print(f"[警告] 未能载入 {gru_ckpt_path}, 将跳过 LSTM_GRU 对比。")
            ok_gru = False

        if not (ok_lstm or ok_gru):
            print("无可用模型，跳过对比...")
            continue

        # ========== 分别做推理 ==========
        if ok_lstm:
            preds_lstm, targets = inference(lstm_model, test_loader, y_scaler, device=device)
        else:
            preds_lstm, targets = None, None

        if ok_gru:
            preds_lstm_gru, _ = inference(lstm_gru_model, test_loader, y_scaler, device=device)
        else:
            preds_lstm_gru = None

        # ========== 计算指标 ==========
        def calc_metrics(pred, true):
            mse = np.mean((pred - true) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred - true))
            return mse, rmse, mae

        if ok_lstm:
            mse_lstm, rmse_lstm, mae_lstm = calc_metrics(preds_lstm, targets)
            print(f"[LSTM]     MSE={mse_lstm:.4f}, RMSE={rmse_lstm:.4f}, MAE={mae_lstm:.4f}")
        else:
            mse_lstm, rmse_lstm, mae_lstm = (None, None, None)

        if ok_gru:
            mse_gru, rmse_gru, mae_gru = calc_metrics(preds_lstm_gru, targets)
            print(f"[LSTM_GRU] MSE={mse_gru:.4f}, RMSE={rmse_gru:.4f}, MAE={mae_gru:.4f}")
        else:
            mse_gru, rmse_gru, mae_gru = (None, None, None)

        # 如果只有一个模型成功加载，则无需对比曲线；若两个都成功则画对比图
        if ok_lstm and ok_gru:
            # 对比曲线
            plt.figure(figsize=(10, 5))
            plt.plot(targets, label='True', color='#4D4D4D', linewidth=2)
            plt.plot(preds_lstm, label='LSTM', color='#6BAED6', linewidth=2)
            plt.plot(preds_lstm_gru, label='LSTM_GRU', color='#74C476', linewidth=2)

            plt.xlabel("Test Sample Index")
            plt.ylabel("能耗值")
            plt.title(f"模型预测对比 - {station}")
            plt.legend()
            compare_fig = f"./分析图/compare_pred_{station}.png"
            plt.savefig(compare_fig, dpi=300, bbox_inches='tight')
            plt.show()

            # 对比指标的柱状图
            metrics_lstm = [mse_lstm, rmse_lstm, mae_lstm]
            metrics_gru = [mse_gru, rmse_gru, mae_gru]
            metric_names = ['MSE', 'RMSE', 'MAE']

            x = np.arange(len(metric_names))
            bar_width = 0.35

            plt.figure(figsize=(8, 5))
            plt.bar(x - bar_width/2, metrics_lstm, width=bar_width, label='LSTM', color='#6BAED6')
            plt.bar(x + bar_width/2, metrics_gru, width=bar_width, label='LSTM_GRU', color='#74C476')

            plt.xticks(x, metric_names)
            plt.ylabel("Metric Value")
            plt.title(f"指标柱状对比 - {station}")
            plt.legend()
            bar_fig = f"./分析图/bar_metrics_{station}.png"
            plt.savefig(bar_fig, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"站点 {station} 测试完成。结果图已保存：\n  {compare_fig}\n  {bar_fig}\n\n")
        else:
            # 若只有一个模型可用，就只画单条预测 vs 真值
            if ok_lstm:
                plt.figure(figsize=(10, 5))
                plt.plot(targets, label='True', color='gray')
                plt.plot(preds_lstm, label='LSTM', color='blue')
                plt.xlabel("Test Sample Index")
                plt.ylabel("能耗值")
                plt.title(f"LSTM 模型预测 - {station}")
                plt.legend()
                single_fig = f"./分析图/lstm_only_{station}.png"
                plt.savefig(single_fig, dpi=300, bbox_inches='tight')
                plt.show()
            if ok_gru:
                plt.figure(figsize=(10, 5))
                plt.plot(targets, label='True', color='gray')
                plt.plot(preds_lstm_gru, label='LSTM_GRU', color='green')
                plt.xlabel("Test Sample Index")
                plt.ylabel("能耗值")
                plt.title(f"LSTM_GRU 模型预测 - {station}")
                plt.legend()
                single_fig = f"./分析图/lstm_gru_only_{station}.png"
                plt.savefig(single_fig, dpi=300, bbox_inches='tight')
                plt.show()

if __name__ == "__main__":
    main_test()
