import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import numpy as np
from utils import create_gnn_encoder, Weighted_Summation
from decoders import get_decoder, get_mlp_encoder, GNNEncoder


class DFCDM(BaseModel):
    def __init__(self, config):
        super(DFCDM, self).__init__(config)
        # Define encoder networks
        self.encoder_GNN = GNNEncoder(layer=config['encoder_type'],
                                      in_channels=config['out_channels'],
                                      hidden_channels=config['out_channels'],
                                      out_channels=config['out_channels']).to(config['device'])

        self.attn_S = Weighted_Summation(config['out_channels'], attn_drop=0.2).to(self.device)
        self.attn_E = Weighted_Summation(config['out_channels'], attn_drop=0.2).to(self.device)
        self.attn_K = Weighted_Summation(config['out_channels'], attn_drop=0.2).to(self.device)
        self.encoder_student_llm = get_mlp_encoder(in_channels=config['in_channels_llm'],
                                                   out_channels=config['out_channels']).to(config['device'])

        self.encoder_exercise_llm = get_mlp_encoder(in_channels=config['in_channels_llm'],
                                                    out_channels=config['out_channels']).to(config['device'])

        self.encoder_knowledge_llm = get_mlp_encoder(in_channels=config['in_channels_llm'],
                                                     out_channels=config['out_channels']).to(config['device'])

        self.encoder_student_init = get_mlp_encoder(in_channels=config['in_channels_init'],
                                                    out_channels=config['out_channels']).to(config['device'])

        self.encoder_exercise_init = get_mlp_encoder(in_channels=config['in_channels_init'],
                                                     out_channels=config['out_channels']).to(config['device'])

        self.encoder_knowledge_init = get_mlp_encoder(in_channels=config['in_channels_init'],
                                                      out_channels=config['out_channels']).to(config['device'])

        self.decoder = get_decoder(config).to(config['device'])

    def mask_nodes(self, x_init, ratio=0.2):
        total_rows = self.config['stu_num'] + self.config['prob_num'] + self.config['know_num']
        mask_rows = np.random.choice(total_rows, int(ratio * total_rows), replace=False)
        x_init[mask_rows] = 0
        return x_init, mask_rows

    def get_data(self, mode='train'):
        if mode == 'train':
            return self.config['train_data'].x_llm, self.config['train_data'].x_init, self.config[
                'train_data'].edge_index
        else:
            if self.config['split'] != 'Stu' and self.config['split'] != 'Exer':
                return self.config['train_data'].x_llm, self.config['train_data'].x_init, self.config[
                    'train_data'].edge_index
            else:
                return self.config['full_data'].x_llm, self.config['full_data'].x_init, self.config[
                    'full_data'].edge_index

    def get_x(self, x_llm, x_init, edge_index):
        if self.config['mode'] == 0:
            student_factor = self.encoder_student_init(x_init[:self.config['stu_num'], ])
            exercise_factor = self.encoder_exercise_init(
                x_init[self.config['stu_num']:self.config['stu_num'] + self.config['prob_num'], ])
            knowledge_factor = self.encoder_knowledge_init(x_init[self.config['stu_num'] + self.config['prob_num']:, ])
        elif self.config['mode'] == 1:
            student_factor = self.encoder_student_llm(x_llm[:self.config['stu_num'], ])
            exercise_factor = self.encoder_exercise_llm(
                x_llm[self.config['stu_num']:self.config['stu_num'] + self.config['prob_num'], ])
            knowledge_factor = self.encoder_knowledge_llm(x_llm[self.config['stu_num'] + self.config['prob_num']:, ])
        elif self.config['mode'] == 2:
            student_factor_init = self.encoder_student_init(x_init[:self.config['stu_num'], ])
            exercise_factor_init = self.encoder_exercise_init(
                x_init[self.config['stu_num']:self.config['stu_num'] + self.config['prob_num'], ])
            knowledge_factor_init = self.encoder_knowledge_init(
                x_init[self.config['stu_num'] + self.config['prob_num']:, ])
            student_factor_llm = self.encoder_student_llm(x_llm[:self.config['stu_num'], ])
            exercise_factor_llm = self.encoder_exercise_llm(
                x_llm[self.config['stu_num']:self.config['stu_num'] + self.config['prob_num'], ])
            knowledge_factor_llm = self.encoder_knowledge_llm(
                x_llm[self.config['stu_num'] + self.config['prob_num']:, ])
            student_factor = self.attn_S([student_factor_init, student_factor_llm])
            exercise_factor = self.attn_E([exercise_factor_init, exercise_factor_llm])
            knowledge_factor = self.attn_K([knowledge_factor_init, knowledge_factor_llm])

        final_x = torch.cat([student_factor, exercise_factor, knowledge_factor], dim=0)

        if self.training:
            x_mask, mask_rows = self.mask_nodes(final_x, ratio=0.2)
            return self.encoder_GNN.forward(x_mask, edge_index), mask_rows
        else:
            return self.encoder_GNN.forward(final_x, edge_index), None

    def forward(self, student_id, exercise_id, knowledge_point, mode='train'):
        x_llm, x_init, edge_index = self.get_data(mode)
        rep, _ = self.get_x(x_llm, x_init, edge_index)
        return self.decoder.forward(rep, student_id, exercise_id, knowledge_point)

    def get_mastery_level(self, mode='eval'):
        x_llm, x_init, edge_index = self.get_data(mode)
        rep, _ = self.get_x(x_llm, x_init, edge_index)
        return self.decoder.get_mastery_level(rep)

    def monotonicity(self):
        self.decoder.monotonicity()
