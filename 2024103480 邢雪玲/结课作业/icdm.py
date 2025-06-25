import dgl
import torch
import numpy as np
import torch.nn as nn
from base import BaseModel
from decoders import Positive_MLP, SAGENet, Weighted_Summation
from utils import NoneNegClipper, dgl2tensor, get_subgraph, concept_distill
import scipy.sparse as sp


class ICDM(BaseModel):
    def __init__(self, config):
        super(ICDM, self).__init__(config)
        self.stu_emb = nn.Embedding(config['stu_num'], config['dim']).to(self.device)
        self.exer_emb_right = nn.Embedding(config['prob_num'], config['dim']).to(self.device)
        self.exer_emb_wrong = nn.Embedding(config['prob_num'], config['dim']).to(self.device)
        self.know_emb = nn.Embedding(config['know_num'], config['dim']).to(self.device)


        self.S_E_right = SAGENet(dim=config['dim'], type=config['agg_type'], device=self.device, layers_num=config['khop'],
                                 d_1=config['d_1'], d_2=config['d_2']).to(self.device)
        self.S_E_wrong = SAGENet(dim=config['dim'], type=config['agg_type'], device=self.device, layers_num=config['khop'],
                                 d_1=config['d_1'], d_2=config['d_2']).to(self.device)
        self.E_C_right = SAGENet(dim=config['dim'], type=config['agg_type'], device=self.device, layers_num=config['khop'],
                                 d_1=config['d_1'], d_2=config['d_2']).to(self.device)
        self.E_C_wrong = SAGENet(dim=config['dim'], type=config['agg_type'], device=self.device, layers_num=config['khop'],
                                 d_1=config['d_1'], d_2=config['d_2']).to(self.device)
        self.S_C = SAGENet(dim=config['dim'], type=config['agg_type'], device=self.device, layers_num=config['khop'], d_1=config['d_1'],
                           d_2=config['d_2']).to(self.device)

        self.attn_S = Weighted_Summation(config['dim'], attn_drop=0.2).to(self.device)
        self.attn_E_right = Weighted_Summation(config['dim'], attn_drop=0.2).to(self.device)
        self.attn_E_wrong = Weighted_Summation(config['dim'], attn_drop=0.2).to(self.device)
        self.attn_E = Weighted_Summation(config['dim'], attn_drop=0.2).to(self.device)
        self.attn_C = Weighted_Summation(config['dim'], attn_drop=0.2).to(self.device)

        exer_id = torch.arange(config['prob_num']).to(self.device)
        exer_id_S = exer_id + torch.full(exer_id.shape, config['stu_num']).to(self.device)
        self.train_right_graph = dgl.in_subgraph(self.config['right_old'],
                                                 torch.cat((torch.tensor(config['exist_idx']), exer_id_S.detach().cpu()),
                                                           dim=-1))
        self.train_wrong_graph = dgl.in_subgraph(self.config['wrong_old'],
                                                 torch.cat((torch.tensor(config['exist_idx']), exer_id_S.detach().cpu()),
                                                           dim=-1))

        if config['split'] == 'Stu':
            config['norm_adj_train'], config['norm_adj_full'] = self.create_adj_mat(config['np_train_old'],
                                                                                config['np_train'])
        else:
            config['norm_adj_train'], config['norm_adj_full'] = self.create_adj_mat(config['np_train'],
                                                                                config['np_train'])

        concept_id = torch.arange(config['know_num']).to(self.device)
        concept_id_S = concept_id + torch.full(concept_id.shape, config['stu_num']).to(self.device)
        self.train_I = dgl.in_subgraph(self.config['I_old'],
                                       torch.cat((torch.tensor(config['exist_idx']), concept_id_S.detach().cpu()), dim=-1))

        self.Involve_Matrix = dgl2tensor(self.config['involve'])[:config['stu_num'], config['stu_num']:].to(self.device)
        self.transfer_stu_layer = nn.Linear(config['dim'], config['know_num']).to(self.device)
        self.transfer_exer_layer = nn.Linear(config['dim'], config['know_num']).to(self.device)
        self.transfer_concept_layer = nn.Linear(config['dim'], config['know_num']).to(self.device)

        self.change_latent_stu = nn.Linear(config['dim'], 16).to(self.device)
        self.change_latent_exer = nn.Linear(config['dim'], 16).to(self.device)

        self.fn_1 = nn.Linear(config['dim'] * 2, config['dim'])
        self.fn_2 = nn.Linear(config['dim'] * 2, config['dim'])

        self.disc_emb = nn.Embedding(config['prob_num'], 1).to(self.device)

        self.positive_mlp = Positive_MLP(config).to(self.device)

        for index, (name, param) in enumerate(self.named_parameters()):
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def compute(self, norm_adj_tmp, emb):
        all_emb = emb
        embs = [emb]
        for layer in range(self.config['gcn_layers']):
            all_emb = torch.sparse.mm(norm_adj_tmp, all_emb)
            embs.append(all_emb)
        out_embs = torch.mean(torch.stack(embs, dim=1), dim=1)
        return out_embs

    def forward(self, student_id, exercise_id, knowledge_point, mode='train'):
        concept_id = torch.where(knowledge_point != 0)[1].to(self.device)
        concept_id_S = concept_id + torch.full(concept_id.shape, self.config['stu_num']).to(self.device)
        concept_id_E = concept_id + torch.full(concept_id.shape, self.config['prob_num']).to(self.device)
        exer_id_S = exercise_id + torch.full(exercise_id.shape, self.config['stu_num']).to(self.device)

        subgraph_node_id_Q = torch.cat((exercise_id.detach().cpu(), concept_id_E.detach().cpu()), dim=-1)
        subgraph_node_id_R = torch.cat((student_id.detach().cpu(), exer_id_S.detach().cpu()), dim=-1)
        subgraph_node_id_I = torch.cat((student_id.detach().cpu(), concept_id_S.detach().cpu()), dim=-1)

        Q_subgraph = get_subgraph(self.config['Q'], subgraph_node_id_Q, device=self.device)

        R_subgraph_Right_train = get_subgraph(self.train_right_graph, subgraph_node_id_R, device=self.device)
        R_subgraph_Wrong_train = get_subgraph(self.train_wrong_graph, subgraph_node_id_R, device=self.device)
        I_subgraph_train = get_subgraph(self.train_I, subgraph_node_id_I, device=self.device)


        if mode != 'train':
            R_subgraph_Right_all = get_subgraph(self.config['right_eval'], subgraph_node_id_R, device=self.device)
            R_subgraph_Wrong_all = get_subgraph(self.config['wrong_eval'], subgraph_node_id_R, device=self.device)
            I_subgraph_all = get_subgraph(self.config['I_eval'], subgraph_node_id_I, device=self.device)

        exer_info_right, exer_info_wrong, concept_info = self.exer_emb_right.weight, self.exer_emb_wrong.weight, self.know_emb.weight
        E_C_right = torch.cat([exer_info_right, concept_info]).to(self.device)
        E_C_wrong = torch.cat([exer_info_wrong, concept_info]).to(self.device)

        E_C_info_right = self.E_C_right(Q_subgraph, E_C_right)
        E_C_info_wrong = self.E_C_wrong(Q_subgraph, E_C_wrong)
        #
        stu_info = self.stu_emb.weight
        S_C = torch.cat([stu_info, concept_info]).to(self.device)
        S_E_right = torch.cat([stu_info, exer_info_right]).to(self.device)
        S_E_wrong = torch.cat([stu_info, exer_info_wrong]).to(self.device)

        if mode == 'train':
            S_E_info_right, S_E_info_wrong = self.S_E_right(R_subgraph_Right_train, S_E_right), self.S_E_wrong(
                R_subgraph_Wrong_train, S_E_wrong)
            S_C_info = self.S_C(I_subgraph_train, S_C)
        else:
            S_E_info_right, S_E_info_wrong, S_E_info_right_all, S_E_info_wrong_all = self.S_E_right(
                R_subgraph_Right_train, S_E_right), \
                self.S_E_wrong(R_subgraph_Wrong_train, S_E_wrong), self.S_E_right(R_subgraph_Right_all,
                                                                                  S_E_right), self.S_E_wrong(
                R_subgraph_Wrong_all, S_E_wrong)
            S_C_info, S_C_info_all = self.S_C(I_subgraph_train, S_C), self.S_C(I_subgraph_all, S_C)

        E_forward_right = self.attn_E_right.forward(
            [E_C_info_right[:self.config['prob_num']], S_E_info_right[self.config['stu_num']:]])

        E_forward_wrong = self.attn_E_wrong.forward(
            [E_C_info_wrong[:self.config['prob_num']], S_E_info_wrong[self.config['stu_num']:]])

        C_forward = self.attn_C.forward(
            [E_C_info_right[self.config['prob_num']:], E_C_info_wrong[self.config['prob_num']:], S_C_info[self.config['stu_num']:]])

        E_forward = E_forward_right * E_forward_wrong

        if mode == 'train':
            S_forward = self.attn_S.forward(
                [S_E_info_right[:self.config['stu_num']], S_E_info_wrong[:self.config['stu_num']],
                 S_C_info[:self.config['stu_num']]]
            )
        else:
            S_forward = self.attn_S.forward(
                [S_E_info_right_all[:self.config['stu_num']], S_E_info_wrong_all[:self.config['stu_num']],
                 S_C_info_all[:self.config['stu_num']]]
            )

        emb = torch.cat([S_forward, E_forward]).to(self.device)
        disc = torch.sigmoid(self.disc_emb(exercise_id))

        def irf(theta, a, b, D=1.702):
            return torch.sigmoid(torch.mean(D * a * (theta - b), dim=1)).to(self.device).view(-1)

        if self.config['cdm_type'] == 'glif':
            if mode == 'train':
                out = self.compute(self.config['norm_adj_train'], emb)
            else:
                out = self.compute(self.config['norm_adj_full'], emb)

            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                out[:self.config['stu_num']]), self.transfer_exer_layer(out[self.config['stu_num']:]), self.transfer_concept_layer(
                C_forward)

            exer_concept_distill = concept_distill(knowledge_point, C_forward)

            state = disc * (torch.sigmoid(S_forward[student_id] * exer_concept_distill) - torch.sigmoid(
                E_forward[exercise_id] * exer_concept_distill)) * knowledge_point
            return self.positive_mlp(state).view(-1)


        elif self.config['cdm_type'] == 'ncdm':

            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                S_forward), self.transfer_exer_layer(E_forward), self.transfer_concept_layer(C_forward)
            state = disc * (torch.sigmoid(S_forward[student_id]) - torch.sigmoid(
                E_forward[exercise_id])) * knowledge_point
            return self.positive_mlp(state).view(-1)

        elif self.config['cdm_type'] == 'mirt':

            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                S_forward), self.transfer_exer_layer(E_forward), self.transfer_concept_layer(C_forward)
            return irf(S_forward[student_id], disc, E_forward[exercise_id])
        else:
            raise ValueError('We do not support it yet')


    def monotonicity(self):
        none_neg_clipper = NoneNegClipper()
        for layer in self.positive_mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)


    def get_mastery_level(self, mode='eval'):
        R_subgraph_Right = self.config['right_eval'].to(self.device)
        R_subgraph_Wrong = self.config['wrong_eval'].to(self.device)
        I_subgraph = self.config['I_eval'].to(self.device)
        Q_subgraph = self.config['Q'].to(self.device)

        exer_info_right = self.exer_emb_right.weight
        exer_info_wrong = self.exer_emb_wrong.weight
        concept_info = self.know_emb.weight

        E_C_right = torch.cat([exer_info_right, concept_info]).to(self.device)
        E_C_wrong = torch.cat([exer_info_wrong, concept_info]).to(self.device)

        E_C_info_right = self.E_C_right(Q_subgraph, E_C_right)
        E_C_info_wrong = self.E_C_wrong(Q_subgraph, E_C_wrong)
        #
        stu_info = self.stu_emb.weight
        S_C = torch.cat([stu_info, concept_info]).to(self.device)
        S_E_right = torch.cat([stu_info, exer_info_right]).to(self.device)
        S_E_wrong = torch.cat([stu_info, exer_info_wrong]).to(self.device)
        S_E_info_right = self.S_E_right(R_subgraph_Right, S_E_right)
        S_E_info_wrong = self.S_E_wrong(R_subgraph_Wrong, S_E_wrong)

        self.attn_S = self.attn_S.to(self.device)
        self.attn_C = self.attn_C.to(self.device)
        self.attn_E_right = self.attn_E_right.to(self.device)
        self.attn_E_wrong = self.attn_E_wrong.to(self.device)

        self.norm_adj_full = self.config['norm_adj_full'].to(self.device)
        self.transfer_stu_layer = self.transfer_stu_layer.to(self.device)
        self.transfer_exer_layer = self.transfer_exer_layer.to(self.device)
        self.transfer_concept_layer = self.transfer_concept_layer.to(self.device)

        E_forward_right = self.attn_E_right.forward(
            [E_C_info_right[:self.config['prob_num']], S_E_info_right[self.config['stu_num']:]])
        E_forward_wrong = self.attn_E_wrong.forward(
            [E_C_info_wrong[:self.config['prob_num']], S_E_info_wrong[self.config['stu_num']:]])
        E_forward = E_forward_right * E_forward_wrong

        S_C_info = self.S_C(I_subgraph, S_C)
        C_forward = self.attn_C.forward(
            [E_C_info_right[self.config['prob_num']:], E_C_info_wrong[self.config['prob_num']:], S_C_info[self.config['stu_num']:]])
        S_forward = self.attn_S.forward(
            [S_E_info_right[:self.config['stu_num']], S_E_info_wrong[:self.config['stu_num']], S_C_info[:self.config['stu_num']]])

        emb = torch.cat([S_forward, E_forward]).to(self.device)

        if self.config['cdm_type'] == 'glif':
            if mode == 'eval':
                out = self.compute(self.config['norm_adj_full'], emb)
            else:
                out = self.compute(self.config['norm_adj_train'], emb)
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                out[:self.config['stu_num']]), self.transfer_exer_layer(out[self.config['stu_num']:]), self.transfer_concept_layer(
                C_forward)
            stu_concept_distill = concept_distill(self.Involve_Matrix, C_forward)
            return torch.sigmoid(S_forward * stu_concept_distill).detach().cpu().numpy()
        else:
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                S_forward), self.transfer_exer_layer(E_forward), self.transfer_concept_layer(
                C_forward)
            return torch.sigmoid(S_forward).detach().cpu().numpy()


    @staticmethod
    def get_adj_matrix(tmp_adj):
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    @staticmethod
    def sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

    def create_adj_mat(self, np_train, np_train_new):
        n_nodes = self.config['stu_num'] + self.config['prob_num']
        train_stu = np_train[:, 0]
        train_exer = np_train[:, 1]
        full_stu = np.vstack((np_train_new))[:, 0]
        full_exer = np.vstack((np_train_new))[:, 1]

        ratings_full = np.ones_like(full_stu, dtype=np.float64)
        ratings_train = np.ones_like(train_stu, dtype=np.float64)

        tmp_adj_full = sp.csr_matrix((ratings_full, (full_stu, full_exer + self.config['stu_num'])), shape=(n_nodes, n_nodes))
        tmp_adj_train = sp.csr_matrix((ratings_train, (train_stu, train_exer + self.config['stu_num'])),
                                      shape=(n_nodes, n_nodes))

        return self.sp_mat_to_sp_tensor(self.get_adj_matrix(tmp_adj_train)).to(self.device), self.sp_mat_to_sp_tensor(self.get_adj_matrix(tmp_adj_full)).to(self.device)
