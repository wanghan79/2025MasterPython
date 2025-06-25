import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def load_model(model, path):
    try:
        model.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')
        return True
    except Exception as e:
        print(f'Error loading model: {e}')
        return False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    