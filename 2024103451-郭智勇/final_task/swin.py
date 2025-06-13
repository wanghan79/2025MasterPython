# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu

# window_size=to_2tuple(self.window_size)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from FPN import fpn


#投影切割的包
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.nn.utils.rnn import pad_sequence


#mask可视化的包
# import matplotlib.pyplot as plt
# import seaborn as sns
# import random
import torch.nn as nn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=312):
        super().__init__()
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x, positions):
        # x: (B, seq_len, dim)
        # positions: (B, seq_len) 包含每个位置的索引（如 0~783）
        pos_emb = self.position_embedding(positions)  # (B, seq_len, dim)
        pos_emb = pos_emb.to(device)
        return x + pos_emb

def create_batch(patch_features_list):
    """
    输入: 多个样本的分块特征列表，每个样本的分块数可变
    输出: 填充后的批量特征 (batch_size, max_seq_len, dim)
    """
    # 展平所有分块的序列维度
    flattened = [feat.view(-1, 768) for feat in patch_features_list]
    # 动态填充
    padded = pad_sequence(flattened, batch_first=True)  # (batch_size, max_seq_len, 768)
    return padded


#绝对位置编码
class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, window_size, num_heads):
        super(AbsolutePositionEmbedding, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        #创建位置编码
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_heads, window_size[0] * window_size[1], window_size[0] * window_size[1])
        )

    def forward(self, x):
        # 将位置编码添加到输入嵌入中
        return x + self.position_embedding

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


#多重向量机
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



def window_partition(x, window_size): #划分窗口
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # print('x.shape=',x.shape)
    #(B,m,window_size,n,window_size,C)
    Wh, Ww = window_size
    x = x.view(B, H // Wh, Wh, W // Ww, Ww, C)
    #(B,m,n,window_size,window_size,C)有m*n个窗口
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, Wh, Ww, C)
    # print('window.shape=',windows.shape)
    return windows


def window_reverse(windows, window_size, H, W): #window_partition的逆操作，将窗口还原为原图像
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    Wh, Ww = window_size
    B = int(windows.shape[0] / (H * W / (Wh * Ww)))
    x = windows.view(B, H // Wh, W // Ww, Wh, Ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim  # 初始化维度
        self.eps = eps  # 初始化epsilon值，防止除零错误
        self.elementwise_affine = elementwise_affine  # 是否使用元素级仿射变换
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))  # 如果使用元素级仿射变换，则初始化权重参数
        else:
            self.register_parameter('weight', None)  # 否则不注册权重参数

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 计算RMS归一化

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)  # 应用RMS归一化并恢复原始数据类型
        if self.weight is not None:
            output = output * self.weight  # 如果有权重参数，则应用权重
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'  # 返回额外的表示信息

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class WindowAttention(nn.Module): #窗口注意力类
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, img_size=(96,128),qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,layer_idx=0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.img_size=img_size

        # 生成旋转后的相对坐标差
        coords_h = torch.arange(window_size[0])  
        coords_w = torch.arange(window_size[1]) 
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 

        # 坐标逆向变换
        dx_rot = relative_coords[1, :, :].float()  
        dy_rot = -relative_coords[0, :, :].float() 

        # 转换为极坐标（基于原始竖排方向）
        r = torch.sqrt(dx_rot ** 2 + dy_rot ** 2 + 1e-9)
        theta = torch.atan2(dx_rot, dy_rot)  


        # 归一化
        max_r = math.sqrt((self.window_size[0] - 1) ** 2 + (self.window_size[1] - 1) ** 2)
        r_norm = r / max_r
        theta_norm = (theta + math.pi) / (2 * math.pi)  # 映射到[0,1]

        # 注册为缓冲张量
        self.register_buffer("polar_input", torch.stack([r_norm, theta_norm], dim=-1))  # [H_rot*W_rot, H_rot*W_rot, 2]

        # 方向感知位置编码MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, num_heads)
        )

        # 创建绝对位置编码
        # self.absolute_pos_embed = AbsolutePositionEmbedding(window_size, num_heads)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv2=nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lambda_init = lambda_init_fn(layer_idx)
        # 将 self.relative_position_bias_table 张量的值初始化为标准差为 0.02 的截断正态分布  原swin
        # trunc_normal_(self.polar_input, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.k = 0

        # Init λ across heads
        self.lambda_init = lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(self.head_dim, eps=1e-9, elementwise_affine=True)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #（3,B,num_head,N,C // num_head）
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        qk2=self.qkv2(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2 = qk2[0], qk2[1]

        #q×k转置然后除以根号下dk，同时进行差分操作

        q = q * self.scale
        q2 = q2 * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn2=(q2 @ k2.transpose(-2, -1))
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn=attn-lambda_full*attn2


        # 处理位置编码
        polar_input = self.polar_input.view(-1, 2)  # [N*N, 2]
        position_bias = self.pos_mlp(polar_input)  # [N*N, num_heads]
        position_bias = position_bias.view(N, N, self.num_heads)  # [N, N, num_heads]
        position_bias = position_bias.permute(2, 0, 1)


        attn = attn + position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)


        x = (attn @ v)
        x = self.subln(x)
        x = (x * (1 - self.lambda_init)).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str: #不重要的函数
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):  #不重要的函数
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=(0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= max(self.window_size):
            self.shift_size = (0, 0)
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            # dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size[0], window_size[1], 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size[0], -self.shift_size[1], self.window_size[0], self.window_size[1])
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)

        # reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size[0], self.shift_size[1],
                                               self.window_size[0], self.window_size[1])
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        # print('x.shape=',x.shape)
        # print('PatchMerging前的图像维度',x.shape)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C #每隔一个高度取一个元素，每隔一个宽度取一个元素
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2),
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=4, in_chans=4, embed_dim=128, norm_layer=None):
        super().__init__()
        img_size = img_size
        # patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution #(56,56)
        self.num_patches = patches_resolution[0] * patches_resolution[1]#3136

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size).to(device)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim).to(device)
        else:
            self.norm = None

    def forward(self, x):
        x=x.to(device)
        B, C, H, W= x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(
                self, img_size=(96,768), patch_size=(4,4), in_chans=3, num_classes=0, #img_size=(H,W)
                 embed_dim=96, depths=[2,4,6,2], num_heads=[4,8,16,32],
                 window_size=(7,8), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches#56 56
        patches_resolution = self.patch_embed.patches_resolution #3136
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        feature_list =[]
        for index,layer in enumerate(self.layers):
            feature_list.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C (1 49 768)
        return x,feature_list

    def forward(self, x):
        x,features_list= self.forward_features(x)
        x = self.head(x)
        return x,features_list

#切割图片
def process_image(img, ratio_range=(0.1, 5), target_size=(),index=0):
    # 预处理图像
    img=np.array(img) #从PIL格式转为Numpy格式
    if img is None:
        print(f"无法读取图像文件：{img}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 垂直投影
    v_projection = np.sum(thresh, axis=0)

    # 找到切割点
    cut_points = []
    start = 0
    for i in range(len(v_projection)):
        if v_projection[i] > 0 and start == 0:
            start = i
        elif v_projection[i] == 0 and start != 0:
            cut_points.append((start - 2, i + 2))
            start = 0
    # 如果最后一列有像素，则添加切割点
    if start != 0:
        cut_points.append((start, len(v_projection)))

    segments = []
    segments_width=[]
    for v_idx, (start, end) in enumerate(cut_points):
        if start<0:#避免有的start为-1
            start=0
        width = end - start
        if width < 10:
            continue
        segment = img[:, start:end] #每个单词都上下加2px空白作为缓冲
        height, width = segment.shape[:2]
        aspect_ratio = width / height if height != 0 else float('inf')
        # 检查宽高比是否在指定范围内
        if ratio_range[0] <= aspect_ratio:
            index+=1
            segments_width.append(width)
            # if index<10000:
            #     # 将NumPy数组转换为PIL图像
            #     pil_image = Image.fromarray(segment)
            #     pil_image.save(f'/home/rootroot/tmp/ZOAaoxJ5vz/数据集/手写体投影切割/{index}.png')
            #     print(f'已保存第{index}张图片到路径')
            segment_resized = cv2.resize(segment, target_size)
            segment_resized= Image.fromarray(cv2.cvtColor(segment_resized, cv2.COLOR_BGR2RGB))
            segments.append(segment_resized)
        else:
            continue
    return segments

#特征图连接
def concatenate_features(features_list):
    # 将每个小块的序列展平并拼接
    global_seq = []
    for patch_feature in features_list:  # 按空间顺序遍历小块
        global_seq.append(patch_feature.squeeze(0))  # (seq_len_patch, dim)
    global_seq = torch.cat(global_seq, dim=0)  # (M*N*seq_len_patch, dim)
    return global_seq.unsqueeze(0)  # (1, global_seq_len, dim)


# 定义图像变换，这里包括将PIL图像转换为tensor的操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # transforms.Resize((64, 832)),
    # 随机擦除 (Random Erasing)
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.03)),  # 注意 scale 参数范围
    # 随机旋转 (Random Rotation)
    transforms.RandomRotation(degrees=(-3, 3)),
    # 仿射变换 (Random Affine)
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
    # 高斯模糊 (Gaussian Blur)
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))# 调整 sigma 范围
])

test_transform=transforms.Compose([
    transforms.ToTensor(), # 将PIL Image或numpy.ndarray转换为tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
#对每一块应用swin,返回每一块的特征图并存储在列表中
def apply_swin_to_segments(segments, swin_model,id):
    features = []
    for segment in segments:
        # 将图像块转换为PIL图像并调整为Tensor
        # pil_image = Image.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
        tensor = transform(segment).unsqueeze(0)  # 添加batch维度
        #应用swin model
        feature = swin_model(tensor)
        features.append(feature)
    return features

#生成自学习权重
class AdaptiveWeightFusion(nn.Module):
    def __init__(self, num_maps):
        super(AdaptiveWeightFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_maps))  # 初始化权重为1

    def forward(self, feature_maps):
        weights = F.softmax(self.weights, dim=0)  # 归一化权重
        weights = weights.view(len(feature_maps), 1, 1)  # 调整权重形状以广播到特征序列
        weighted_maps = feature_maps * weights.to(device)  # 加权求和
        final_feature = torch.sum(weighted_maps, dim=0)  # 形状为 (seq_length, dim)
        return final_feature

class Swin(nn.Module):
    def __init__(self, swin_model=swin_model):
        super().__init__()
        self.swin_model = swin_model
        self.layernorm=nn.LayerNorm(768)
        self.index=0
        self.id=0

    def forward(self, x,ratio_range=(0.0, 5),target_size=(256,224),type='train'): #img,阈值，最终尺寸
    # def forward(self, x,ratio_range=(0.0, 2),target_size=(128,128),type='train'): #img,阈值，最终尺寸
        maps=[]
        # print('每次输入到swin的x为：',len(x))
        for item in x:
            #切割每张图片
            segments= process_image(item, ratio_range, target_size, index=self.index)
            # self.index+=len(segments)
            # self.index+=len(weights)

            #从这一行到if全是我的猜测
            #将存储切割图片的列表转换为tensor
            if type=='train':
                segments_tensor = [transform(image) for image in segments]
            else:
                segments_tensor = [test_transform(image) for image in segments]

            # 保存数据增强后的图像
            # for item in segments_tensor:
            #     # 将图像块转换为PIL图像并调整为Tensor
            #     # 如果 item 是一个 PyTorch Tensor，需要先转换为 NumPy 数组
            #     if isinstance(item, torch.Tensor):
            #         # 将 Tensor 转换为 NumPy 数组，并调整维度顺序（从 C,H,W 到 H,W,C）
            #         item = item.permute(1, 2, 0).numpy()
            #     # 确保 item 是 NumPy 数组
            #     if isinstance(item, np.ndarray):
            #         # 检查数据类型并转换
            #         if item.dtype == np.float32 or item.dtype == np.float64:
            #             # 将 float 数据缩放到 [0, 255] 并转换为 uint8
            #             item = (item * 255).astype(np.uint8)
            #         # 将 BGR 转换为 RGB
            #         rgb_image = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            #         # 转换为 PIL 图像
            #         pil_image = Image.fromarray(rgb_image)
            #     pil_image.save(f'/home/rootroot/tmp/ZOAaoxJ5vz/数据集/数据增强后的图像/{self.id+1}.png')
            #     print(f'保存{self.id+1}张图片')
            #     self.id+=1

            if len(segments_tensor)==0:
                item.save('/home/rootroot/tmp/ZOAaoxJ5vz/数据集/未切割图片/未切割.png')
                print('存在未切割')
            segments_tensor = torch.stack(segments_tensor)

            #得到切割图片的特征图
            feature_map,feature_list=self.swin_model(segments_tensor)

            # feature_map=fpn(feature_list) #多层次特征融合
            num_maps=len(feature_map)


            # === 绝对位置编码 ===
            # 定义绝对位置编码（正弦余弦位置编码）
            seq_length=feature_map[0].shape[0]
            dim=feature_map[0].shape[1]
            absolute_position_encoding = torch.zeros(feature_map[0].shape[0], feature_map[0].shape[1], device=feature_map.device)
            pos = torch.arange(0, seq_length, device=feature_map.device).float().unsqueeze(1)
            _2i = torch.arange(0, dim, step=2, device=feature_map.device).float()

            absolute_position_encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / dim)))
            absolute_position_encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / dim)))

            # 将绝对位置编码扩展到与特征图相同的形状
            absolute_position_encoding = absolute_position_encoding.unsqueeze(0)  # 形状为 (1, seq_length, dim)
            absolute_position_encoding = absolute_position_encoding.expand(num_maps, -1, -1)  # 形状为 (n, seq_length, dim)

            # 定义相对位置编码（简单的位置索引编码）

            position_encodings = torch.arange(num_maps).unsqueeze(1).unsqueeze(2) # 形状为 [8, 1, 1]

            # 将相对位置编码扩展到与特征图相同的形状
            position_encodings = position_encodings.expand(num_maps, feature_map[0].shape[0], feature_map[0].shape[1]).to(device)

            # 将相对位置编码添加到每张特征图
            encoded_maps = feature_map + position_encodings+absolute_position_encoding # 形状为 (n, seq_length, dim)

            # 使用自适应权重融合
            fusion_layer = AdaptiveWeightFusion(num_maps=num_maps)
            final_feature = fusion_layer(encoded_maps)
            maps.append(final_feature)
        mask=None
        maps=torch.stack(maps, dim=0)
        return maps,mask

swin=Swin().to(device)


