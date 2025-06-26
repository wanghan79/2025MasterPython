import torch

def sinkhorn(log_alpha, n_iters=20):
    for _ in range(n_iters):
        log_alpha = log_alpha - log_alpha.logsumexp(dim=1, keepdim=True)
        log_alpha = log_alpha - log_alpha.logsumexp(dim=2, keepdim=True)
    return log_alpha.exp()