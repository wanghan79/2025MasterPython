import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import random
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


###########################################
# 1. 定义 InteractionDataset - 用于 DataLoader
###########################################
class InteractionDataset(Dataset):
    """
    将 (user, item, rating) 列表封装成 PyTorch Dataset 的简易示例。
    """
    def __init__(self, data_list, user_key, item_key, rating_key):
        # data_list 里的每条数据形如 (u, i, r)
        # user_key, item_key, rating_key 分别对应 ESSRec 里的 self.USER_ID, self.ITEM_ID, self.RATING
        self.records = []
        self.user_key = user_key
        self.item_key = item_key
        self.rating_key = rating_key

        for (u, i, r) in data_list:
            rec = {self.user_key: u, self.item_key: i, self.rating_key: r}
            self.records.append(rec)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    @staticmethod
    def collate_fn(batch):
        result = defaultdict(list)
        for b in batch:
            for k,v in b.items():
                result[k].append(v)
        # 将 user_id, item_id 转成 long; rating 转成 float
        out = {}
        for k, vals in result.items():
            if "rating" in k:
                out[k] = torch.tensor(vals, dtype=torch.float32)
            else:
                out[k] = torch.tensor(vals, dtype=torch.long)
        return out


###########################################
# 2. ESSRec模型 (融合高阶聚合 + 对比学习)
###########################################
class ESSRec(nn.Module):
    def __init__(self, dataset, args):
        super(ESSRec, self).__init__()

        self.USER_ID = "user_id:token"
        self.ITEM_ID = "item_id:token"
        self.RATING = "rating:float"

        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        print(f"Number of users: {self.n_users}, Number of items: {self.n_items}")

        self.interaction_matrix = dataset.interaction_matrix.astype(np.float32)
        self.net_matrix = dataset.net_matrix

        print(f"Interaction matrix shape: {self.interaction_matrix.shape}")
        print(f"Net matrix shape: {self.net_matrix.shape}")

        # 给符号网络加对角（I）
        diag = np.ones(self.n_users)
        I = sp.coo_matrix(sp.diags(diag), dtype=np.float32)
        self.net_matrix = self.net_matrix + I
        self.net_matrix = sp.coo_matrix(self.net_matrix)

        # 从 args 中获取参数
        self.embedding_size = args.embedding_size
        self.gnn_layer_size = args.gnn_layer_size       # 原本多层卷积层数
        self.gnn_layer_size_k = args.gnn_layer_size_k   # 新增: 再叠加多阶聚合
        self.sim_threshold = args.sim_threshold
        self.device = args.device
        self.criterion = nn.MSELoss()

        # 多阶卷积矩阵
        self.ACM = self.get_convolution_matrix(self.interaction_matrix).to(self.device)
        self.ACM_T = self.get_convolution_matrix(self.interaction_matrix.transpose()).to(self.device)
        self.SCM = self.get_convolution_matrix(self.net_matrix).to(self.device)

        # Embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size).to(self.device)

        # 一些线性层
        self.sim_layer = nn.Linear(self.embedding_size * 2, 2).to(self.device)
        self.user_concat = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)
        self.item_concat = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)
        self.concat_user_item_layer = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)
        self.concat_PON_layer = nn.Linear(self.embedding_size * 3, self.embedding_size).to(self.device)
        self.concat_PN_layer = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)
        self.concat_user_layer = nn.Linear(self.embedding_size * 2, self.embedding_size).to(self.device)

        # Contrastive 超参
        self.contrast_margin = getattr(args, "contrast_margin", 1.0)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def forward(self):
        # U-I图上的卷积
        user_init_embedding, item_init_embedding = self.user_item_graph_convolution()
        # item 进一步聚合
        item_rep = torch.sparse.mm(self.ACM, item_init_embedding.to(self.device))

        # 拼接
        E = self.concat_user_item_layer(torch.cat([user_init_embedding, item_rep], dim=1))

        # 对符号网络做多阶聚合
        # 先第一次
        P_total, N_total, O_total = None, None, None
        current_E = E
        for k in range(self.gnn_layer_size_k):
            P, N, O = self.graph_convolution(self.net_matrix, current_E, item_rep)
            if P_total is None:
                P_total = P
                N_total = N
                O_total = O
            else:
                P_total += P
                N_total += N
                O_total += O
            # 可以更新 current_E = (P + N + O)/3 ...
            current_E = (P + N + O)/3

        # 取平均
        P = P_total / self.gnn_layer_size_k
        N = N_total / self.gnn_layer_size_k
        O = O_total / self.gnn_layer_size_k

        user_pn_embedding = self.concat_PN_layer(torch.cat([P, N], dim=1))
        user_final_embedding = self.concat_user_layer(torch.cat([user_pn_embedding, O], dim=1))
        item_final_embedding = item_init_embedding

        return user_final_embedding, item_final_embedding

    def user_item_graph_convolution(self):
        item_rep = torch.sparse.mm(self.ACM, self.item_embedding.weight.to(self.device))
        user_rep = torch.sparse.mm(self.ACM_T, self.user_embedding.weight.to(self.device))

        item_init_embedding = self.item_concat(
            torch.cat([self.item_embedding.weight.to(self.device),
                       torch.sparse.mm(self.ACM_T, item_rep)], dim=1))

        user_init_embedding = self.user_concat(
            torch.cat([self.user_embedding.weight.to(self.device),
                       torch.sparse.mm(self.ACM, user_rep)], dim=1))
        return user_init_embedding, item_init_embedding

    def graph_convolution(self, sparse_matrix, E, item_rep):
        # P,N,O 拆分
        P, N, O = self.get_signed_split_matrices(sparse_matrix, item_rep)
        PCM = self.get_convolution_matrix(P).to(self.device)
        OCM = self.get_convolution_matrix(O).to(self.device)
        NCM = self.get_convolution_matrix(N).to(self.device)

        p_now, n_now, o_now = None, None, None
        for i in range(self.gnn_layer_size):
            if i == 0:
                p_now = torch.mm(PCM, E)
                n_now = torch.mm(NCM, E)
                o_now = torch.mm(OCM, E)
            else:
                p1 = torch.mean(torch.cat([torch.mm(PCM, p_now).unsqueeze(0),
                                           torch.mm(NCM, n_now).unsqueeze(0)], dim=0), dim=0)
                n1 = torch.mean(torch.cat([torch.mm(NCM, p_now).unsqueeze(0),
                                           torch.mm(PCM, n_now).unsqueeze(0)], dim=0), dim=0)
                o1 = torch.mean(torch.cat([
                    torch.mm(PCM, o_now).unsqueeze(0),
                    torch.mm(NCM, o_now).unsqueeze(0),
                    torch.mm(OCM, p_now).unsqueeze(0),
                    torch.mm(OCM, n_now).unsqueeze(0),
                    torch.mm(OCM, o_now).unsqueeze(0)
                ], dim=0), dim=0)
                p_now, n_now, o_now = p1, n1, o1

        return p_now, n_now, o_now

    def get_signed_split_matrices(self, sparse_matrix, item_rep):
        row = torch.LongTensor(sparse_matrix.row).to(self.device)
        col = torch.LongTensor(sparse_matrix.col).to(self.device)
        data = sparse_matrix.data

        source_rep = item_rep[row]
        target_rep = item_rep[col]
        cos_sim = F.cosine_similarity(source_rep, target_rep).cpu().detach().numpy()

        sim_index = np.argwhere(cos_sim > self.sim_threshold).squeeze(axis=1)
        diff_index = np.argwhere(cos_sim <= self.sim_threshold).squeeze(axis=1)
        trust_index = np.argwhere(data == 1).squeeze(axis=1)
        distrust_index = np.argwhere(data == -1).squeeze(axis=1)
        total = np.union1d(trust_index, distrust_index)

        p_index = np.intersect1d(trust_index, sim_index)
        n_index = np.intersect1d(distrust_index, diff_index)
        o_index = np.setdiff1d(total, np.union1d(p_index, n_index))

        P = self.get_split_matrix(sparse_matrix, p_index)
        N = self.get_split_matrix(sparse_matrix, n_index)
        O = self.get_split_matrix(sparse_matrix, o_index)
        return P, N, O

    def get_split_matrix(self, sparse_matrix, index):
        row = sparse_matrix.row
        col = sparse_matrix.col
        row = row[index]
        col = col[index]
        data = np.ones(len(index))
        matrix = sp.coo_matrix((data, (row, col)), shape=(self.n_users, self.n_users))
        return matrix

    def get_convolution_matrix(self, matrix):
        # mean聚合
        D = self.get_inverse_degree_matrix(matrix)
        L = D * matrix
        L = sp.coo_matrix(L)
        return self.get_sparse_tensor(L)

    def get_inverse_degree_matrix(self, matrix):
        sumArr = (matrix > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -1)
        return sp.coo_matrix(sp.diags(diag), dtype=np.float32)

    def get_sparse_tensor(self, sparse_matrix):
        # 将 sparse_matrix.row, .col, .data 转换为 numpy 数组
        row = np.array(sparse_matrix.row)
        col = np.array(sparse_matrix.col)
        indices = np.vstack((row, col))
        data = np.array(sparse_matrix.data)
        # 使用 torch.sparse_coo_tensor
        i = torch.LongTensor(indices)
        d = torch.FloatTensor(data)
        return torch.sparse_coo_tensor(i, d, sparse_matrix.shape, device=self.device)


###########################################
# 3. 额外：对比学习损失(用户-用户)
###########################################
def contrastive_loss_intra_view(user_embeddings, net_matrix, margin=1.0, sample_size=512):
    """
    在同一个视图内, 正边用户相似(1-cos_sim), 负边用户相斥(ReLU(cos_sim+margin))
    """
    row = net_matrix.row
    col = net_matrix.col
    data = net_matrix.data

    indices = np.arange(len(row))
    if len(indices) == 0:
        return torch.tensor(0.0, device=user_embeddings.device)
    sample_size = min(sample_size, len(indices))
    sampled = np.random.choice(indices, size=sample_size, replace=False)

    row_s = row[sampled]
    col_s = col[sampled]
    sign_s = data[sampled]

    loss_c = 0.0
    for i in range(sample_size):
        u1 = row_s[i]
        u2 = col_s[i]
        s = sign_s[i]
        emb1 = user_embeddings[u1]
        emb2 = user_embeddings[u2]
        cos_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).mean()

        if s > 0:
            loss_c += (1.0 - cos_sim)  # 希望 cos_sim→1
        else:
            val = cos_sim + margin
            loss_c += F.relu(val)
    return loss_c / sample_size


###########################################
# 4. 训练循环(含进度条) + 评估(RMSE, MAE)
###########################################
def train_model(dataset, args, train_data, test_data, epochs=10, batch_size=256, alpha=0.5):
    """
    示例：利用 DataLoader + tqdm 进行训练并评估
    alpha: 对比损失在总损失中的权重
    """
    model = ESSRec(dataset, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 构造 dataset & dataloader
    train_dataset = InteractionDataset(train_data, model.USER_ID, model.ITEM_ID, model.RATING)
    test_dataset = InteractionDataset(test_data, model.USER_ID, model.ITEM_ID, model.RATING)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              collate_fn=InteractionDataset.collate_fn)
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False,
                             collate_fn=InteractionDataset.collate_fn)

    # 训练
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        # batch 级别进度条
        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch") as tloader:
            for batch_data in tloader:
                rating_gt = batch_data[model.RATING].to(args.device)
                scores = model.predict(batch_data)
                mse_loss = F.mse_loss(scores, rating_gt)
                
                # Contrastive loss: user embedding
                with torch.no_grad():
                    user_emb, _ = model.forward()
                c_loss = contrastive_loss_intra_view(user_emb, dataset.net_matrix, margin=model.contrast_margin)
                
                loss = (1-alpha)*mse_loss + alpha*c_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # 评估
        rmse_val, mae_val = evaluate_model(model, test_loader, args)
        print(f"[Epoch {epoch}/{epochs}]  Loss={avg_loss:.4f}  RMSE={rmse_val:.4f}  MAE={mae_val:.4f}")

    return model


def evaluate_model(model, test_loader, args):
    model.eval()
    preds = []
    gts = []

    with torch.no_grad():
        for batch_data in test_loader:
            rating_gt = batch_data[model.RATING].to(args.device)
            scores = model.predict(batch_data)
            preds.append(scores.cpu())
            gts.append(rating_gt.cpu())

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    rmse = torch.sqrt(torch.mean((preds - gts)**2))
    mae = torch.mean(torch.abs(preds - gts))

    return rmse.item(), mae.item()


import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from tqdm import tqdm
import random


class EpinionsDataset:
    """
    Epinions 数据集加载和预处理类，适配 ESSRec 模型的要求。
    """

    def __init__(self, path_inter="epinions.inter", path_net="epinions.net", train_ratio=0.8):
        # 用户和物品的 ID 映射表
        self.user_map = {}
        self.item_map = {}
        self.user_cnt = 0
        self.item_cnt = 0

        # 用户-物品评分数据：格式 {user_id: [(item_id, rating), ...]}
        self.user_item_dict = defaultdict(list)

        # 用户-用户符号社交网络：格式 {(u1, u2): sign}
        self.user_user_sign = {}

        # 加载交互数据和社交网络数据
        self._load_inter(path_inter)
        self._load_net(path_net)

        # 构建评分矩阵 (interaction_matrix) 和符号网络矩阵 (net_matrix)
        self.interaction_matrix = self._build_interaction_matrix()
        self.net_matrix = self._build_net_matrix()

        # 划分训练集和测试集
        self.train_data, self.test_data = self._train_test_split(ratio=train_ratio)

    def _map_user_id(self, old_u):
        """用户 ID 重映射"""
        if old_u not in self.user_map:
            self.user_map[old_u] = self.user_cnt
            self.user_cnt += 1
        return self.user_map[old_u]

    def _map_item_id(self, old_i):
        """物品 ID 重映射"""
        if old_i not in self.item_map:
            self.item_map[old_i] = self.item_cnt
            self.item_cnt += 1
        return self.item_map[old_i]

    def _load_inter(self, path_inter):
        """
        读取 epinions.inter 文件，构建 user-item 数据字典。
        格式: user_id item_id rating
        """
        print(f"Loading interaction data from {path_inter}...")
        num_lines = sum(1 for _ in open(path_inter, "r"))

        with open(path_inter, "r") as f:
            for line in tqdm(f, total=num_lines, desc="Reading interaction file"):
                # 跳过可能的表头和空行
                if line.startswith("user_id") or line.strip() == "":
                    continue

                # 尝试解析每行数据
                try:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue

                    old_u, old_i, r = parts
                    old_u = int(old_u)
                    old_i = int(old_i)
                    r = float(r)

                    new_u = self._map_user_id(old_u)
                    new_i = self._map_item_id(old_i)
                    self.user_item_dict[new_u].append((new_i, r))

                except ValueError as e:
                    print(f"Skipping invalid line: {line.strip()} - Error: {e}")
                except Exception as e:
                    print(f"Unexpected error when reading line: {line.strip()} - Error: {e}")

    def _load_net(self, path_net):
        """
        读取 epinions.net 文件，构建用户-用户符号网络字典。
        格式: user_id1 user_id2 sign
        """
        print(f"Loading social network data from {path_net}...")
        num_lines = sum(1 for _ in open(path_net, "r"))

        with open(path_net, "r") as f:
            for line in tqdm(f, total=num_lines, desc="Reading social network file"):
                # 跳过可能的表头和空行
                if line.startswith("source_id") or line.strip() == "":
                    continue

                # 尝试解析每行数据
                try:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue

                    old_u1, old_u2, sign = parts
                    old_u1 = int(old_u1)
                    old_u2 = int(old_u2)
                    sign = int(sign)  # 1 或 -1

                    # 映射为连续 ID
                    new_u1 = self._map_user_id(old_u1)
                    new_u2 = self._map_user_id(old_u2)

                    self.user_user_sign[(new_u1, new_u2)] = sign

                except ValueError as e:
                    print(f"Skipping invalid line: {line.strip()} - Error: {e}")
                except Exception as e:
                    print(f"Unexpected error when reading line: {line.strip()} - Error: {e}")


    def _build_interaction_matrix(self):
        """
        构建用户-物品评分矩阵（稀疏格式）。
        行：用户，列：物品，值：评分。
        """
        print("Constructing interaction matrix...")
        rows, cols, data = [], [], []
        for u, interactions in tqdm(self.user_item_dict.items(), desc="Building interaction matrix"):
            for i, r in interactions:
                rows.append(u)
                cols.append(i)
                data.append(r)
        # 构建稀疏矩阵
        interaction_matrix = sp.coo_matrix((data, (rows, cols)),
                                           shape=(self.user_cnt, self.item_cnt),
                                           dtype=np.float32)
        return interaction_matrix

    def _build_net_matrix(self):
        """
        构建用户-用户符号社交网络的邻接矩阵。
        行列：用户，值：±1。
        """
        print("Constructing net matrix...")
        rows, cols, data = [], [], []
        for (u1, u2), sign in tqdm(self.user_user_sign.items(), desc="Building net matrix"):
            rows.append(u1)
            cols.append(u2)
            data.append(sign)
        # 构建用户-用户网络矩阵
        net_matrix = sp.coo_matrix((data, (rows, cols)),
                                   shape=(self.user_cnt, self.user_cnt),
                                   dtype=np.float32)
        return net_matrix

    def _train_test_split(self, ratio=0.8):
        """
        将用户-物品评分数据集划分为训练集和测试集。
        """
        print("Splitting dataset into train and test sets...")
        all_ratings = []
        for u, interactions in self.user_item_dict.items():
            for i, r in interactions:
                all_ratings.append((u, i, r))
        random.shuffle(all_ratings)

        # 划分数据集
        split_idx = int(len(all_ratings) * ratio)
        train_data = all_ratings[:split_idx]
        test_data = all_ratings[split_idx:]
        return train_data, test_data

    def num(self, field):
        """
        返回字段对应的计数：用户或物品数。
        """
        if field == "user_id:token":
            return self.user_cnt
        elif field == "item_id:token":
            return self.item_cnt
        else:
            raise ValueError(f"Unknown field: {field}")


#############################################
# 6. 主程序：实例化数据集 & 开始训练
#############################################
if __name__ == "__main__":
    # 文件路径
    inter_path = "/home/wangzf/work/pycharmproject/signedsocialrec1/dataset/epinions/epinions.inter"
    net_path = "/home/wangzf/work/pycharmproject/signedsocialrec1/dataset/epinions/epinions.net"

    # 实例化数据集
    dataset = EpinionsDataset(inter_path, net_path)

    print(f"Dataset summary:")
    print(f" - Users: {dataset.num('user_id:token')}")
    print(f" - Items: {dataset.num('item_id:token')}")
    print(f" - Interaction matrix shape: {dataset.interaction_matrix.shape}")
    print(f" - Net matrix shape: {dataset.net_matrix.shape}")
    print(f" - Train interactions: {len(dataset.train_data)}")
    print(f" - Test interactions: {len(dataset.test_data)}")


    # 调用 ESSRec 模型
    class Args:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        embedding_size = 16
        gnn_layer_size = 2
        gnn_layer_size_k = 2
        sim_threshold = 0.2
        lr = 1e-3
        contrast_margin = 1.0


    args = Args()

    # 开始训练模型
    model = train_model(dataset, args, dataset.train_data, dataset.test_data,
                        epochs=50, batch_size=40960, alpha=0.5)

    print("Training completed successfully!")
