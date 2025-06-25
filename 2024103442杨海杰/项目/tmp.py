import torch
import numpy as np
import pandas as pd
import os
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(0)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# file = "./train/soc-sign-bitcoinotc.csv"
# out_file = "./train/soc-sign-bitcoinotc-reidx.txt"
# seed = 6666
MAXINT = np.iinfo(np.int64).max
# length = 50
# max_hop = 7
# n_layers = 16
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
#
# torch.manual_seed(seed)

# # 将原始数据集节点id进行新的映射
# def reidx(file, out_file):
#     data = pd.read_csv(file, delimiter=",")
#     dataset = torch.tensor(np.array(data), dtype=torch.int).to(device)
#     all_nodes = torch.unique(dataset[:, : 2])
#
#     reidx_dict = {nodes_id.item(): new_id for new_id, nodes_id in enumerate(all_nodes)}
#
#     data["src"] = data["src"].map(reidx_dict)
#     data["dst"] = data["dst"].map(reidx_dict)
#
#     res = np.array(data)
#     np.savetxt(out_file, res, delimiter=",", fmt="%d")


def load_data(file):

    # if not os.path.exists(out_file):
    #     reidx(file, out_file)

    dataset = np.loadtxt(file, delimiter=" ")
    dataset = torch.tensor(dataset, dtype=torch.int)
#数据集第三列的数据大于等于4置为1，小于4置为-1
    dataset[dataset[:, 2] >= 4, 2] = 1
    dataset[dataset[:, 2] < 4, 2] = -1

    # shuffle
    shuffle_idx = torch.randperm(dataset.shape[0])
    dataset = dataset[shuffle_idx]

    # # split
    # edge_nums = dataset.shape[0]
    # train_data = dataset[: int(edge_nums*0.8)].T.to(device)
    # test_data = dataset[int(edge_nums*0.8):].T

    # sort the train_data to non-decrease order by src
    # train_data = train_data[:, torch.sort(train_data[0]).indices]
    train_data = dataset[dataset[:, 0].argsort()]#把原数据按第一列排序

    return train_data


def index2adj(train_data, max_node_num):
    edge_index = train_data[: 2] #取前两列
    val = train_data[2]  #取第三列
    # node_num = torch.unique(edge_index).shape[0]
    # max_node_num = torch.max(edge_index).item()

    adj_signed = torch.sparse_coo_tensor(edge_index, val, (max_node_num, max_node_num))
    adj_signed = adj_signed.to_dense()
    #稀疏矩阵(只存储非零元素(索引+值))————>稠密矩阵(存储所有元素(包括0)) 就是生成邻接矩阵

    adj_unsigned = adj_signed.clone()
    adj_unsigned[adj_unsigned == -1] = 1

    return adj_unsigned, adj_signed, edge_index, val


def degree_offset(max_node_num, adj_unsigned):
    # degree
    out_degree = adj_unsigned.sum(1)#计算每个结点的出度

    # offset
    offset = torch.zeros(max_node_num, dtype=torch.int)
    for i in range(max_node_num-1):
        offset[i+1] = offset[i] + out_degree[i]
#offset表示每个节点在 CSR 格式中的起始位置。举例：max_node_num = 4，out_degree = [2, 1, 3, 0]，offset[0] = 0
                                                                                    #offset[1] = offset[0] + out_degree[0] = 0 + 2 = 2
                                                                                    #offset[2] = offset[1] + out_degree[1] = 2 + 1 = 3
                                                                                    #offset[3] = offset[2] + out_degree[2] = 3 + 3 = 6
    return out_degree, offset


def generate_data(file, max_node_num):

    train_data = load_data(file) #load_data返回train_data,这里处理了第三列大于4为1，小于4为-1，并且按第一列排序

    adj_unsigned, adj_signed, train_edge_index, train_val, = index2adj(train_data, max_node_num)
    #无符号邻接矩阵，有符号邻接矩阵，边的索引（2，节点数），边+1或-1
    degree, offset = degree_offset(max_node_num, adj_unsigned)#出度，CSR格式的起始位置

    return train_edge_index, train_val, adj_unsigned, adj_signed, degree, offset


def repeat(x, n_layers):
    x = torch.unsqueeze(x, dim=0)
    return torch.cat([x for _ in range(n_layers)])


def get_edge_sign(node1, node2, sign_adj):
    return torch.gather(sign_adj[node1], 1, node2.unsqueeze(-1)).squeeze(-1)


def get_next_node(cur_node, out_degree, adj_offset, choice, edge_index):
    # out degree 0 was set to 1
    node_degree = torch.gather(out_degree, 1, cur_node) - 1
    node_adj_offset = torch.gather(adj_offset, 1, cur_node)
    chosen_edge = choice % torch.clamp(node_degree, min=1) + node_adj_offset
    next_node = torch.gather(edge_index[:, 1], 1, chosen_edge)
    return next_node.squeeze(-1), chosen_edge.squeeze(-1), \
           node_degree.squeeze(-1), node_adj_offset.squeeze(-1)


def genWalk(train_edge_index, adj_signed, degree, offset, max_node_num, length = 50, max_hop = 7, n_layers = 16, directed = False):
    edge_index_rp = repeat(train_edge_index, n_layers)  # (n_layers, 2, edges)
                        #把边的结构沿着列方向复制了16次，得到（2，n_layers*edges)
    #edge_index_rp = train_edge_index.unsqueeze(0).repeat(n_layers, 1, 1)

    node_idx_selected_pool = torch.unique(train_edge_index)  # the node exist in dataset
    node_num = torch.tensor(node_idx_selected_pool.shape[0])
    node_num_rp = repeat(node_num, n_layers)

    out_degree_rp = repeat(degree, n_layers)
    adj_offset_rp = repeat(offset, n_layers)

    # output
    walk_nodes = torch.zeros([length+1, n_layers], dtype=torch.int64, device=device)
    walk_edges = torch.zeros([length, n_layers], dtype=torch.int64, device=device)
    walk_signs = torch.zeros([length, n_layers], dtype=torch.int64, device=device)
    spatial_pos = (max_hop + 1) * torch.ones([n_layers, max_node_num, max_node_num], dtype=torch.int64, device=device)
    #spatial_pos形状[n_layers, max_node_num, max_node_num]，保存每个层内所有节点对之间的“距离信息”（跳数乘上边符号）
    #初始化时每个位置都是max_hop + 1即8
    choices = torch.randint(0, MAXINT, [length+1, n_layers], device=device)
    #为每个路径的每一步生成一个随机整数，用于采样和决策
    # start nodes随机选择起始节点
    walk_nodes[0] = choices[0] % node_num_rp

    # if node with out degree = 0 is selected, choose 0 to start
    nodes_degree_filter = torch.gather(out_degree_rp, 1, walk_nodes[0].unsqueeze(-1)) != 1
    walk_nodes[0] = walk_nodes[0] * nodes_degree_filter.squeeze(-1)

    # sample the seconde nodes
    walk_nodes[1], walk_edges[0], _, _ = get_next_node(
        walk_nodes[0].unsqueeze(-1), out_degree_rp, 
        adj_offset_rp, choices[1].unsqueeze(-1), edge_index_rp)

    walk_signs[0] = get_edge_sign(walk_nodes[0], walk_nodes[1], adj_signed)

    for i in range(1, length):
        next_node, chosen_edge, node_degree, node_adj_offset = \
                get_next_node(walk_nodes[i].unsqueeze(-1), out_degree_rp, adj_offset_rp, choices[i + 1].unsqueeze(-1), edge_index_rp)
        # non-backtracking
        chosen_edge += walk_nodes[i - 1] == next_node
        chosen_edge = (chosen_edge - node_adj_offset) % torch.clamp(node_degree, min=1) + node_adj_offset
        walk_nodes[i + 1] = torch.gather(edge_index_rp[:, 1], 1, chosen_edge.unsqueeze(-1)).squeeze(-1)
        walk_edges[i] = chosen_edge
        walk_signs[i] = get_edge_sign(walk_nodes[i], walk_nodes[i + 1], adj_signed)
    #从walk_nodes[0]走到walk_nodes[49]
    walk_nodes = walk_nodes.T
    walk_edges = walk_edges.T
    walk_signs = walk_signs.T

    layers_iter = torch.arange(n_layers, dtype=torch.int64, device=device)

    # random walk spatial position
    for d in range(max_hop, -1, -1):

        for i in range(length - d + 1):

            # compute sign
            sign = torch.ones(n_layers)
            for j in range(i, i+d):
                sign = sign.mul(walk_signs[layers_iter, j])
                #对路径 walk_nodes[i] → walk_nodes[i+1] → ... → walk_nodes[i+d] 的所有边符号做连乘

            spatial_pos[layers_iter, walk_nodes[:, i], walk_nodes[:, i + d]] = (sign * (d + 1)).long()
            if not directed: #如果是无向图，则对称赋值（(u,v) 和 (v,u) 相同）
                spatial_pos[layers_iter, walk_nodes[:, i + d], walk_nodes[:, i]] = (sign * (d + 1)).long()

    return spatial_pos



# if __name__ == "__main__":
#
#     train_edge_index, train_val, adj_unsigned, adj_signed, degree, offset, max_node_num = generate_data(file, out_file)
#
#     spatial_pos = genWalk(train_edge_index, adj_signed, degree, offset, max_node_num, length, max_hop, n_layers)
#
#     # feature_file =  "./data/bitcoinotc-spatial_pos"
#     #
#     # torch.save(spatial_pos, feature_file)
#     #
#     # feature = torch.load(feature_file)
#     print(spatial_pos)
#     print(spatial_pos.sum(dim=2).sum(dim=1))


