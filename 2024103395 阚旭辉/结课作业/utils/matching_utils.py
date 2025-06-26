from scipy.optimize import linear_sum_assignment
import torch

def hungarian_matching(cost_matrix):
    cost = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind

def compute_match_score(emb1, emb2):
    cost_matrix = torch.cdist(emb1, emb2)
    _, _ = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    return cost_matrix.mean()