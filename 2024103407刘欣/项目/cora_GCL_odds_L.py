import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import dgl.function as fn  # 导入 DGL 的函数模块
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 文件
import torch  # 导入 PyTorch
import torch.nn.functional as F  # 导入 PyTorch 的函数式 API
import numpy as np  # 导入 numpy
import pandas as pd  # 导入 pandas
from torch_geometric.data import Data  # 导入 PyG 的 Data 类
from ogb.nodeproppred import Evaluator  # 导入 OGB 的评估器
import dgl  # 导入 DGL
from logger import Logger  # 导入日志记录器

"""
    带有边特征的图 Transformer 层
"""

"""
    工具函数
"""


# Product阶段
def src_dot_dst(src_field, dst_field, out_field):
    # 定义一个函数，计算源节点和目标节点特定字段的点积，并将结果存储到边数据中
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}  # 点积操作

    return func


# Scaling阶段   在模型中用注意力分数out_field
def scaling(field, scale_constant):
    # 定义一个函数，对边数据中的特定字段进行缩放
    def func(edges):  # edges是包含边的相关数据
        return {field: ((edges.data[field]) / scale_constant)}  # 缩放操作

    return func  # 以字典形式输出


# 使用显式边特征改进隐式注意力分数（如果可用） Scaling的下一步
def imp_exp_attn(implicit_attn, explicit_edge):
    """
    implicit_attn: K 和 Q 的输出（隐式注意力）
    explicit_edge: 显式边特征
    """

    def func(edges):
        return {
            implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])
        }  # 结合隐式和显式注意力

    return func


# 复制边特征以传递给 FFN_e   加入边后的下一步
def out_edge_features(edge_feat):
    def func(edges):
        return {"e_out": edges.data[edge_feat]}  # 将边特征复制到 'e_out'

    return func


def exp(field):
    # 定义一个函数，对边数据中的特定字段应用指数函数
    def func(edges):
        # 为了 softmax 的数值稳定性，对值进行截断
        return {
            field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))
        }  # 应用指数函数

    return func


"""
    单个注意力头
"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, in_edge_dim):
        super().__init__()  # 调用父类构造函数
        self.out_dim = out_dim  # 输出维度
        self.num_heads = num_heads  # 注意力头数
        if use_bias:
            self.Q = nn.Linear(
                in_dim, out_dim * num_heads, bias=True
            )  # 查询向量线性变换
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)  # 键向量线性变换
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)  # 值向量线性变换
            self.proj_e = nn.Linear(
                in_edge_dim, out_dim * num_heads, bias=True
            )  # 边特征线性变换
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_edge_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        g.apply_edges(src_dot_dst("K_h", "Q_h", "score"))  # 计算注意力分数
        g.apply_edges(scaling("score", np.sqrt(self.out_dim)))  # 缩放分数
        g.apply_edges(imp_exp_attn("score", "proj_e"))  # 融合边特征
        g.apply_edges(out_edge_features("proj_e"))  # 复制边特征
        g.apply_edges(exp("score"))  # softmax 前的指数
        g.update_all(fn.u_mul_e("V_h", "score", "m"), fn.sum("m", "wV"))  # 消息聚合
        g.update_all(fn.copy_e("score", "m"), fn.sum("m", "z"))  # 归一化因子

    def forward(self, g, h, e):
        e = e.float()  # 边特征转为 float
        Q_h = self.Q(h)  # 计算 Q
        K_h = self.K(h)  # 计算 K
        V_h = self.V(h)  # 计算 V
        proj_e = self.proj_e(e)  # 计算边特征投影
        g.ndata["Q_h"] = Q_h.view(-1, self.num_heads, self.out_dim)  # 节点 Q
        g.ndata["K_h"] = K_h.view(-1, self.num_heads, self.out_dim)  # 节点 K
        g.ndata["V_h"] = V_h.view(-1, self.num_heads, self.out_dim)  # 节点 V
        g.edata["proj_e"] = proj_e.view(-1, self.num_heads, self.out_dim)  # 边特征
        self.propagate_attention(g)  # 执行注意力传播
        h_out = g.ndata["wV"] / (
            g.ndata["z"] + torch.full_like(g.ndata["z"], 1e-6)
        )  # 归一化输出
        e_out = g.edata["e_out"]  # 边特征输出
        return h_out, e_out  # 返回节点和边的输出


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        residual=True,
        use_bias=False,
        in_edge_dim=None,
    ):
        super().__init__()  # 调用父类构造函数
        self.in_channels = in_dim  # 输入通道数
        self.out_channels = out_dim  # 输出通道数
        self.num_heads = num_heads  # 注意力头数
        self.dropout = dropout  # dropout 比例
        self.residual = residual  # 是否使用残差
        self.layer_norm = layer_norm  # 是否使用层归一化
        self.batch_norm = batch_norm  # 是否使用批归一化

        if in_edge_dim is None:
            raise ValueError("in_edge_dim must be provided")  # 必须提供边特征维度

        out_dim_per_head = out_dim // num_heads
        self.attention = MultiHeadAttentionLayer(
            in_dim, out_dim_per_head, num_heads, use_bias, in_edge_dim
        )  # 多头注意力层
        self.e_input_proj = nn.Linear(in_edge_dim, out_dim)  # 边特征输入投影

        self.O_h = nn.Linear(out_dim, out_dim)  # 节点输出线性层
        self.O_e = nn.Linear(out_dim, out_dim)  # 边输出线性层

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)  # 第一层节点层归一化
            self.layer_norm1_e = nn.LayerNorm(out_dim)  # 第一层边层归一化
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)  # 第一层节点批归一化
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)  # 第一层边批归一化

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)  # 节点 FFN 第一层
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)  # 节点 FFN 第二层
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)  # 边 FFN 第一层
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)  # 边 FFN 第二层

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)  # 第二层节点层归一化
            self.layer_norm2_e = nn.LayerNorm(out_dim)  # 第二层边层归一化
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)  # 第二层节点批归一化
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)  # 第二层边批归一化

    def forward(self, g, h, e):
        h_in1 = h  # 保存输入节点特征
        e_in1 = e  # 保存输入边特征
        h_attn_out, e_attn_out = self.attention(g, h, e)  # 多头注意力
        h = h_attn_out.view(-1, self.out_channels)  # 调整节点输出形状
        e = e_attn_out.view(-1, self.out_channels)  # 调整边输出形状
        h = F.dropout(h, self.dropout, training=self.training)  # 节点 dropout
        e = F.dropout(e, self.dropout, training=self.training)  # 边 dropout
        h = self.O_h(h)  # 节点输出线性变换
        e = self.O_e(e)  # 边输出线性变换
        if self.residual:
            h = h_in1 + h  # 节点残差连接
            e = self.e_input_proj(e_in1) + e  # 边残差连接
        if self.layer_norm:
            h = self.layer_norm1_h(h)  # 第一层节点层归一化
            e = self.layer_norm1_e(e)  # 第一层边层归一化
        if self.batch_norm:
            h = self.batch_norm1_h(h)  # 第一层节点批归一化
            e = self.batch_norm1_e(e)  # 第一层边批归一化
        h_in2 = h  # 保存节点特征
        e_in2 = e  # 保存边特征
        h = self.FFN_h_layer1(h)  # 节点 FFN 第一层
        h = F.relu(h)  # 激活
        h = F.dropout(h, self.dropout, training=self.training)  # dropout
        h = self.FFN_h_layer2(h)  # 节点 FFN 第二层
        e = self.FFN_e_layer1(e)  # 边 FFN 第一层
        e = F.relu(e)  # 激活
        e = F.dropout(e, self.dropout, training=self.training)  # dropout
        e = self.FFN_e_layer2(e)  # 边 FFN 第二层
        if self.residual:
            h = h_in2 + h  # 节点残差
            e = e_in2 + e  # 边残差
        if self.layer_norm:
            h = self.layer_norm2_h(h)  # 第二层节点层归一化
            e = self.layer_norm2_e(e)  # 第二层边层归一化
        if self.batch_norm:
            h = self.batch_norm2_h(h)  # 第二层节点批归一化
            e = self.batch_norm2_e(e)  # 第二层边批归一化
        return h, e  # 返回节点和边特征


class NodeClassificationMLP(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        list_FC_layers = []
        for i in range(len(layer_dims) - 1):
            list_FC_layers.append(
                nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True)
            )
        self.FC_layers = nn.ModuleList(list_FC_layers)

    def forward(self, x):
        y = x
        for layer in self.FC_layers[:-1]:
            y = layer(y)
            y = F.relu(y)
        embedding = y  # 投影层输出
        output = self.FC_layers[-1](y)
        # 修复：确保输出为二维 [num_nodes, n_classes]
        if output.dim() == 1:
            output = output.unsqueeze(1)
        return embedding, output


class GraphTransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params["in_dim"]  # node_dim (feat is an integer)
        hidden_dim = net_params["hidden_dim"]
        out_dim = net_params["out_dim"]
        n_classes = net_params["n_classes"]
        num_heads = net_params["num_heads"]
        in_feat_dropout = net_params["in_feat_dropout"]
        dropout = net_params["dropout"]
        n_layers = net_params["L"]

        self.readout = net_params["readout"]
        self.layer_norm = net_params["layer_norm"]
        self.batch_norm = net_params["batch_norm"]
        self.residual = net_params["residual"]
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params["device"]
        self.lap_pos_enc = net_params["lap_pos_enc"]
        self.wl_pos_enc = net_params["wl_pos_enc"]
        self.use_embedding = net_params.get("use_embedding", True)
        max_wl_role_index = 100

        if self.lap_pos_enc:
            pos_enc_dim = net_params["pos_enc_dim"]
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        # 判断输入特征类型，决定是否使用 embedding
        if self.use_embedding:
            self.embedding_h = nn.Embedding(
                in_dim_node, hidden_dim
            )  # node feat is an integer
        else:
            self.input_linear = nn.Linear(in_dim_node, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        edge_feat_dim = net_params.get("edge_feat_dim", None)
        if edge_feat_dim is None:
            raise ValueError("edge_feat_dim must be provided in net_params")

        self.layers = nn.ModuleList()
        # 第一层
        self.layers.append(
            GraphTransformerLayer(
                hidden_dim,
                hidden_dim,
                num_heads,
                dropout,
                self.layer_norm,
                self.batch_norm,
                self.residual,
                in_edge_dim=edge_feat_dim,
            )
        )
        # 中间层（如果有）
        for _ in range(1, n_layers):
            self.layers.append(
                GraphTransformerLayer(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    self.layer_norm,
                    self.batch_norm,
                    self.residual,
                    in_edge_dim=hidden_dim,
                )
            )
        # 注意：所有层都用hidden_dim，MLP层负责输出类别数
        self.MLP_layer = NodeClassificationMLP([hidden_dim, n_classes])

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):
        # 判断输入特征类型
        if h.dtype == torch.long and hasattr(self, "embedding_h"):
            h = self.embedding_h(h)
        elif h.dtype == torch.float or h.dtype == torch.float32:
            h = self.input_linear(h)
        else:
            raise ValueError(f"Unsupported input feature dtype: {h.dtype}")

        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            h = h + h_wl_pos_enc

        h = self.in_feat_dropout(h)

        # GraphTransformer Layers
        for conv in self.layers:
            h, e = conv(g, h, e)

        # output
        embedding, output = self.MLP_layer(h)
        return embedding, output  # 明确返回两个对象，避免嵌套元组

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted negative log likelihood loss for unbalanced classes
        criterion = nn.NLLLoss(weight=weight)  # 修改为NLLLoss
        pred = F.log_softmax(pred, dim=-1)  # 新增：log_softmax
        loss = criterion(pred, label)

        return loss


# 对比学习组件
def dropout_edge(edge_index, edge_attr, p=0.2):
    device = edge_index.device
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges, device=device) > p
    return edge_index[:, mask], edge_attr[mask]


def contrastive_loss(emb_orig, emb_aug, temperature=0.5, batch_size=512):
    emb_orig = F.normalize(emb_orig, dim=1)
    emb_aug = F.normalize(emb_aug, dim=1)
    total_loss = torch.tensor(0.0, device=emb_orig.device)
    num_nodes = emb_orig.size(0)
    for i in range(0, num_nodes, batch_size):
        start = i
        end = min(i + batch_size, num_nodes)
        batch_orig = emb_orig[start:end]
        batch_aug = emb_aug[start:end]
        logits = torch.mm(batch_orig, batch_aug.t()) / temperature
        labels = torch.arange(end - start, device=emb_orig.device)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss * (end - start)
    return total_loss / num_nodes


def pretrain(model, data, optimizer, save_path, device):
    model.train()
    optimizer.zero_grad()

    # 原始视图
    src, dst = data.edge_index
    g = dgl.graph((src, dst), num_nodes=data.num_nodes).to(device)
    g.ndata["feat"] = data.x.to(device)
    g.edata["feat"] = data.edge_attr.to(device)
    emb_orig = model(g, g.ndata["feat"], g.edata["feat"])[0]
    if emb_orig.dim() == 1:
        emb_orig = emb_orig.unsqueeze(0)

    # 增强视图
    edge_index_aug, edge_attr_aug = dropout_edge(data.edge_index, data.edge_attr, p=0.2)
    src_aug, dst_aug = edge_index_aug
    g_aug = dgl.graph((src_aug, dst_aug), num_nodes=data.num_nodes).to(device)
    g_aug.ndata["feat"] = data.x.to(device)
    g_aug.edata["feat"] = edge_attr_aug.to(device)
    emb_aug = model(g_aug, g_aug.ndata["feat"], g_aug.edata["feat"])[0]
    if emb_aug.dim() == 1:
        emb_aug = emb_aug.unsqueeze(0)

    # 计算对比损失
    loss = contrastive_loss(emb_orig, emb_aug)

    loss.backward()
    optimizer.step()
    torch.save(model.state_dict(), save_path)
    print(f"Pretrained model saved to {save_path}")


def parse_cora_with_edge_features():
    bert_feat_path = "/home/a6/vis/lx/项目1/results/cora_feat768.csv"
    edge_odds_path = "/home/a6/vis/lx/项目1/results/egge_odds_10.json"

    # 加载节点特征
    df = pd.read_csv(bert_feat_path)
    node_features_list = df["node_feat"].apply(lambda x: [float(i) for i in x.split()])
    data_X = np.array(node_features_list.tolist(), dtype=np.float32)

    # 加载标签和论文ID到索引的映射（关键修改：统一去除引号和空格）
    content = np.genfromtxt(
        "./dataset/cora_orig/cora/cora.content", dtype=np.dtype(str)
    )
    paper_ids = [row[0].strip('"').strip() for row in content]  # 去除引号和空格
    paper_id_to_index = {pid: i for i, pid in enumerate(paper_ids)}

    # 加载边数据（关键修改：过滤自环边和无效节点）
    with open(edge_odds_path, "r") as f:
        edge_data = json.load(f)

    edges, edge_features, edge_types = [], [], []
    skipped = 0
    for edge in edge_data:
        # 统一去除引号和空格（确保和paper_ids格式一致）
        src_pid = str(edge["source"]).strip('"').strip()
        dst_pid = str(edge["target"]).strip('"').strip()

        # 过滤自环边（新增）
        if src_pid == dst_pid:
            skipped += 1
            continue

        # 检查节点是否存在
        if src_pid not in paper_id_to_index:
            print(f"SKIP: 源节点不存在 | PID: '{src_pid}' (原始值: {edge['source']})")
            skipped += 1
            continue
        if dst_pid not in paper_id_to_index:
            print(f"SKIP: 目标节点不存在 | PID: '{dst_pid}' (原始值: {edge['target']})")
            skipped += 1
            continue

        src = paper_id_to_index[src_pid]
        dst = paper_id_to_index[dst_pid]

        # 检查索引范围（新增）
        if src >= len(paper_ids) or dst >= len(paper_ids):
            print(
                f"SKIP: 索引越界 | src={src}(max {len(paper_ids) - 1}), dst={dst}(max {len(paper_ids) - 1})"
            )
            skipped += 1
            continue

        # 处理多关系边
        for rel in edge["relationslist"]:
            if rel == -1:  # 跳过无效关系
                continue
            edges.append([src, dst])
            # 修改：只取edge_odds中的最大值作为边特征
            edge_features.append([max([float(o) for o in edge["edge_odds"]])])
            edge_types.append(rel)

    # 关键断言（新增）
    num_nodes = data_X.shape[0]
    edge_index = np.array(edges, dtype=np.int64).T
    assert (
        edge_index.max() < num_nodes
    ), f"边索引越界: max={edge_index.max()} >= nodes={num_nodes}"
    print(f"边统计 | 有效边: {len(edges)}, 自环边: {skipped}, 总节点: {num_nodes}")

    # 标签处理
    labels = content[:, -1]
    class_map = {
        "Case_Based": 0,
        "Genetic_Algorithms": 1,
        "Neural_Networks": 2,
        "Probabilistic_Methods": 3,
        "Reinforcement_Learning": 4,
        "Rule_Learning": 5,
        "Theory": 6,
    }

    data_Y = np.array([class_map[l] for l in labels], dtype=np.int64)
    return data_X, data_Y, edge_index, edge_features, edge_types, len(class_map)


# 训练函数
def train(model, g, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    embedding, out = model(g, g.ndata["feat"], g.edata["feat"])  # 解包embedding和output
    out = F.log_softmax(out, dim=-1)

    # 修复：保证out为二维
    if out.dim() == 1:
        out = out.unsqueeze(1)

    train_idx = train_idx.to(data.y.device).long()

    # 调试与安全检查
    num_classes = out.shape[1]
    y_train = data.y[train_idx]
    # 新增调试输出
    if num_classes == 1:
        print(
            f"[警告] 输出类别数为1，out.shape={out.shape}, y_train.min={y_train.min()}, y_train.max={y_train.max()}"
        )
    assert (
        y_train.min() >= 0 and y_train.max() < num_classes
    ), f"标签越界: min={y_train.min()}, max={y_train.max()}, num_classes={num_classes}"
    assert y_train.dtype == torch.long, f"标签类型错误: {y_train.dtype}"

    loss = F.nll_loss(out[train_idx], y_train)
    loss.backward()
    optimizer.step()
    return loss.item()


# 测试函数
@torch.no_grad()
def test(model, g, data, split_idx, evaluator):
    model.eval()
    embedding, out = model(g, g.ndata["feat"], g.edata["feat"])  # 解包embedding和output
    out = F.log_softmax(out, dim=-1)  # 新增：log_softmax
    y_pred = out.argmax(dim=-1, keepdim=True)
    y_true = data.y.reshape(-1, 1)
    split_idx = {k: v.to(data.y.device).long() for k, v in split_idx.items()}

    return {
        "train": evaluator.eval(
            {"y_true": y_true[split_idx["train"]], "y_pred": y_pred[split_idx["train"]]}
        )["acc"],
        "valid": evaluator.eval(
            {"y_true": y_true[split_idx["valid"]], "y_pred": y_pred[split_idx["valid"]]}
        )["acc"],
        "test": evaluator.eval(
            {"y_true": y_true[split_idx["test"]], "y_pred": y_pred[split_idx["test"]]}
        )["acc"],
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Relation Cora (RGCN)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--num_bases", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 加载数据
    data_X, data_Y, edge_index, edge_features, edge_types, num_classes = (
        parse_cora_with_edge_features()
    )

    # 新增检查：验证标签范围
    assert (
        np.min(data_Y) >= 0 and np.max(data_Y) < num_classes
    ), f"标签越界: min={np.min(data_Y)}, max={np.max(data_Y)}"  # <--- 新增检查

    data = Data(
        x=torch.tensor(data_X, dtype=torch.float32).to(device),
        y=torch.tensor(data_Y, dtype=torch.long).to(device),
        edge_index=torch.tensor(edge_index, dtype=torch.long).to(device),
        edge_attr=torch.tensor(edge_features, dtype=torch.float32).to(device),
        edge_type=torch.tensor(edge_types, dtype=torch.long).to(device),
    )

    # 新增：将 edge_index 转为 DGLGraph，并赋值特征
    src, dst = data.edge_index
    g = dgl.graph((src, dst), num_nodes=data.x.shape[0]).to(device)
    g.ndata["feat"] = data.x
    g.edata["feat"] = data.edge_attr

    # 数据集划分
    num_nodes = data.x.shape[0]
    perm = torch.randperm(num_nodes, device=device)
    split_idx = {
        "train": perm[: int(num_nodes * 0.6)],
        "valid": perm[int(num_nodes * 0.6) : int(num_nodes * 0.8)],
        "test": perm[int(num_nodes * 0.8) :],
    }

    # 新增检查：验证索引范围
    assert split_idx["train"].max() < num_nodes  # <--- 新增检查
    assert split_idx["valid"].max() < num_nodes
    assert split_idx["test"].max() < num_nodes

    evaluator = Evaluator(name="ogbn-arxiv")
    logger = Logger(args.runs, args)

    # 新增打印和断言
    print(f"num_classes={num_classes}")
    assert num_classes > 1, f"类别数必须大于1，当前num_classes={num_classes}"

    for run in range(args.runs):
        net_params = {
            "in_dim": data.x.shape[1],
            "hidden_dim": args.hidden_channels,
            "out_dim": args.hidden_channels,  # 保持一致
            "n_classes": num_classes,
            "num_heads": args.num_bases,  # 修正参数名
            "in_feat_dropout": args.dropout,
            "dropout": args.dropout,
            "L": args.num_layers,
            "readout": "none",
            "layer_norm": False,
            "batch_norm": True,
            "residual": True,
            "device": device,
            "lap_pos_enc": False,
            "wl_pos_enc": False,
            "pos_enc_dim": 0,
            "use_embedding": False,
            "edge_feat_dim": data.edge_attr.shape[1],
        }

        # 模型初始化
        model = GraphTransformerNet(net_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # 预训练阶段
        print(f"=== Run {run + 1} Pretraining ===")
        for epoch in range(1, args.epochs + 1):
            pretrain(
                model,
                data,
                optimizer,
                save_path=f"pretrain_model_run{run}.pth",
                device=device,
            )

        # 安全加载模型
        try:
            model.load_state_dict(
                torch.load(
                    f"pretrain_model_run{run}.pth",
                    map_location=device,
                    weights_only=True,  # 安全加载参数
                )
            )
        except TypeError:  # 兼容旧版本
            model.load_state_dict(
                torch.load(f"pretrain_model_run{run}.pth", map_location=device)
            )

        # 微调阶段
        print(f"\n=== Run {run + 1} Fine-tuning ===")
        for epoch in range(1, args.epochs + 1):
            loss = train(model, g, data, split_idx["train"], optimizer)
            result = test(model, g, data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                print(
                    f"Run: {run + 1:02d}, "
                    f"Epoch: {epoch:03d}, "
                    f"Loss: {loss:.4f}, "
                    f'Train: {result["train"] * 100:.2f}% '
                    f'Valid: {result["valid"] * 100:.2f}% '
                    f'Test: {result["test"] * 100:.2f}%'
                )

        logger.print_statistics(run)

    logger.print_statistics()


if __name__ == "__main__":
    main()
