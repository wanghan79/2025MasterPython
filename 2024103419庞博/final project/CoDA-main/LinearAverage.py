import torch
from torch.autograd import Function
from torch import nn
import math
import torch.nn.functional as F
from utils.utils import *

class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize,T=0.05, momentum=0.5):
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.momentum = momentum
        self.register_buffer('params', torch.tensor([T, momentum]))
        self.register_buffer('memory', torch.zeros(outputSize, inputSize))
        # self.register_buffer('memory_logits', torch.zeros(outputSize, class_num))
        self.register_buffer('targets_memory', torch.zeros(outputSize, ))
        self.flag = 0
        self.T = T
        self.memory =  self.memory.cuda()
        self.memory_first = True

    def forward(self, x, use_softmax=True):
        out = torch.mm(x, self.memory.t())
        if use_softmax:
            out = out/self.T
        else:
            out = torch.exp(torch.div(out, self.T))
            Z_l = (out.mean() * self.nLem).clone().detach().item()
            # print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            out = torch.div(out, Z_l).contiguous()

        return out

    def update_weight(self, features, index):

        weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
        weight_pos.mul_(self.momentum)
        weight_pos.add_(torch.mul(features.data, 1 - self.momentum))

        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        self.memory.index_copy_(0, index, updated_weight)
        self.memory = F.normalize(self.memory)#.cuda()


    def set_weight(self, features, index):

        self.memory.index_select(0, index.data.view(-1)).resize_as_(features)

    # def update_weight_logits(self, logits, index):
    #     weight_pos = self.memory_logits.index_select(0, index.data.view(-1)).resize_as_(logits)
    #     weight_pos.mul_(self.momentum)
    #     weight_pos.add_(torch.mul(logits.data, 1 - self.momentum))
    #
    #     w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
    #     updated_weight = weight_pos.div(w_norm)
    #     self.memory_logits.index_copy_(0, index, updated_weight)
    #
    #
    # def set_weight_logits(self, logits, index):
    #     self.memory_logits.index_select(0, index.data.view(-1)).resize_as_(logits)