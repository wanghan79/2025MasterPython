from typing import Union, List, Tuple

import numpy as np
import torch.nn
from torch import nn
from torch.nn import functional as F


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def Attn(in_channels, using_sa=True):
    return AttnBlock(in_channels) if using_sa else nn.Identity()

class Phi(nn.Conv1d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h: torch.Tensor):
        return h.mul(1 - self.resi_ratio) + super().forward(h).mul_(self.resi_ratio)

class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi

class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        if K == 4:
            self.ticks = np.linspace(1/3/K, 1-1/3/K, K)
        else:
            self.ticks = np.linspace(1/2/K, 1-1/2/K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Union[Phi, nn.Module]:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'

class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        if K == 4:
            self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)
        else:
            self.ticks = np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Union[Phi, nn.Module]:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 1e-6 else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h, inplace=True)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h, inplace=True)
        h = self.dropout(h)
        h = self.conv2(h)
        return self.nin_shortcut(x) + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels

        self.norm = Normalize(in_channels)
        self.qkv_module = nn.Conv1d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        qkv: torch.Tensor = self.qkv_module(self.norm(x))
        B, _, L = qkv.shape
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, L).unbind(1)

        # 计算注意力
        q = q.view(B, C, L).contiguous()
        q = q.permute(0, 2, 1).contiguous() # B, L, C
        k = k.view(B, C, L).contiguous() # B, C, L
        w = torch.bmm(q, k).mul_(self.w_ratio) # B, L, L   w[B, i, j] = sum_c q[B, i, C]k[B, C, j]
        w = F.softmax(w, dim=2)

        # attend to values
        v = v.view(B, C, L).contiguous()
        w = w.permute(0, 2, 1).contiguous() # B, L, L (first L of k, second L of q)
        h = torch.bmm(v, w) # B, C, L (L of q) h[B, C, j] = sum_i v[B, C, i] w[B, i, j]
        h = h.view(B, C, L).contiguous()

        return x + self.proj_out(h)

class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1), mode='constant', value=0))

class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))

class Encoder(nn.Module):
    def __init__(self, in_channels, z_channels, ch=128,
                 ch_mult=(1, 2, 4, 8), num_res_blocks=2, dropout_rate=0.0,
                 using_sa=True,
                 using_mid_sa=True):
        """

        :param in_channels:
        :param ch:
        :param ch_mult:
        :param num_res_blocks:
        :param dropout_rate:
        :param using_sa: 是否使用自注意力
        :param using_mid_sa: 中间块是否使用自注意力
        """
        super().__init__()
        # 参数
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # 下采样
        self.conv_in = nn.Conv1d(in_channels=in_channels, out_channels=ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1, ) + tuple(ch_mult)

        self.downs = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            res_blocks = nn.ModuleList()
            attn_blocks = nn.ModuleList()

            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                res_blocks.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout_rate=dropout_rate))

                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn_blocks.append(Attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = res_blocks
            down.attn = attn_blocks
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.downs.append(down)

        # middle
        block_in = ch * ch_mult[self.num_resolutions-1]
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout_rate=dropout_rate)
        self.mid.attn_1 = Attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout_rate=dropout_rate)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(in_channels=block_in, out_channels=z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x) -> torch.Tensor:
        # down sampling
        h = self.conv_in(x)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.downs[i_level].block[i_block](h)
                if len(self.downs[i_level].attn) > 0:
                    h = self.downs[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.downs[i_level].downsample(h)

        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))

        return h

class Decoder(nn.Module):
    def __init__(self, in_channels, z_channels, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                 dropout_rate=0.0, using_sa=True, using_mid_sa=True):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # compute in_ch_mult, block_in and curr_res at lowest_res
        in_ch_mult = (1, ) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = nn.Conv1d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout_rate=dropout_rate)
        self.mid.attn_1 = Attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout_rate=dropout_rate)

        # up_sampling
        self.ups = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            res_blocks = nn.ModuleList()
            attn_blocks = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_blocks.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout_rate=dropout_rate))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn_blocks.append(Attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = res_blocks
            up.attn = attn_blocks

            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.ups.insert(0, up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # z to block_in

        # middle
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # up_sampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.ups[i_level].block[i_block](h)
                if len(self.ups[i_level].attn) > 0:
                    h = self.ups[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.ups[i_level].upsample(h)

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h



class VectorQuantizer(nn.Module):
    def __init__(self, vocab_size, z_channels, using_znorm, beta: float=0.25, default_qresi_counts=0,
                 segment_nums=None, quant_resi_ratio=0.5, share_quant_resi=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.Cvae = z_channels
        self.using_znorm = using_znorm
        self.segment_nums = segment_nums

        self.quant_resi_ratio = quant_resi_ratio

        # build quant_resi block
        if share_quant_resi == 0:
            self.quant_resi = PhiNonShared(
                [
                    (Phi(z_channels, quant_resi_ratio) if abs(quant_resi_ratio) > 1e-6 else nn.Identity())
                    for _ in range(default_qresi_counts or len(self.segment_nums))
                ]
            )
        elif share_quant_resi == 1:
            self.quant_resi = PhiShared(
                Phi(z_channels, quant_resi_ratio) if abs(quant_resi_ratio) > 1e-6 else nn.Identity()
            )
        else:
            self.quant_resi = PhiPartiallyShared(
                nn.ModuleList(
                    [
                        (Phi(z_channels, quant_resi_ratio) if abs(quant_resi_ratio) > 1e-6 else nn.Identity())
                        for _ in range(share_quant_resi)
                    ]
                )
            )

        self.register_buffer('ema_vocab_hit_SV', torch.full(size=(len(self.segment_nums), self.vocab_size),
                                                            fill_value=0.0))

        self.record_hit = 0

        self.beta: float = beta

        self.quantize_embedding = nn.Embedding(self.vocab_size, self.Cvae)

    def forward(self, f_BCl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param f_BCl: B代表batch_size, C代表feature channels, l代表 current feature length
        :return:
        """
        if f_BCl.dtype != torch.float32:
            f_BCl = f_BCl.float()

        B, C, L = f_BCl.shape # B: batch_size, C: feature_channels, L: max feature_length

        f_no_grad = f_BCl.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        mean_vq_loss: torch.Tensor = torch.tensor(0.0).to(device=f_BCl.device)
        with torch.amp.autocast('cuda', enabled=False):

            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BCl.device)

            num_scales = len(self.segment_nums) # SN表示一共有多少个尺度

            for scale_idx, segment_num in enumerate(self.segment_nums): # 每个尺度都有该尺度下的patch数量，si表示当前是第si个尺度，sn表示当前尺度下的segment数量

                # 找到最近的嵌入
                if self.using_znorm:
                    rest_NC = (
                        F.interpolate(f_rest, size=segment_num, mode="area").permute(0, 2, 1).reshape(-1, C)
                            if (scale_idx != num_scales - 1)
                            else f_rest.permute(0, 2, 1).reshape(-1, C)
                    )
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.quantize_embedding.weight.data.T, dim=0), dim=1)
                else:
                    raise "TODO"

                hit_V = idx_N.bincount(minlength=self.vocab_size).float()

                idx_Bl = idx_N.view(B, segment_num) # shape: [B, pn] # 当前尺度下的长度为l==pn，原始的而且是最大的长度则是L
                # 上采样后的从量化嵌入重建的本尺度下的特征
                h_BCl = (
                    F.interpolate(input=self.quantize_embedding(idx_Bl).permute(0, 2, 1), size=L,
                                  mode='area').contiguous()
                    if (scale_idx != num_scales - 1)
                    else self.quantize_embedding(idx_Bl).permute(0, 2, 1).contiguous()
                )
                h_BCl = self.quant_resi[scale_idx/(num_scales-1)](h_BCl)
                f_hat = f_hat + h_BCl
                f_rest = f_rest - h_BCl

                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BCl).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

            mean_vq_loss *= 1.0 / num_scales
            f_hat = (f_hat.data - f_no_grad).add_(f_BCl)

        return f_hat, mean_vq_loss

    def embed_to_fhat(self, ms_h_BCl: List[torch.Tensor], last_one=False):
        ls_fhat_BCl = []
        B = ms_h_BCl[0].shape[0]

        # 最后一个尺度下块的数量
        L = self.segment_nums[-1]
        # 有多少个尺度
        num_scales = len(self.segment_nums)

        f_hat = ms_h_BCl[0].new_zeros(B, self.Cvae, L, dtype=torch.float32)
        for si, pn in enumerate(self.segment_nums):
            cur_h_BCl = ms_h_BCl[si]
            if si < num_scales - 1:
                cur_h_BCl = F.interpolate(cur_h_BCl, size=L, mode='area')
            cur_h_BCl = self.quant_resi[si/(num_scales-1)](cur_h_BCl)
            f_hat.add_(cur_h_BCl)
            if last_one:
                ls_fhat_BCl = f_hat
            else: ls_fhat_BCl.append(f_hat.clone())

        return ls_fhat_BCl

    def f_to_idxBl_or_fhat(self, f_BCl:torch.Tensor, to_fhat:bool):
        B, C, L = f_BCl.shape
        z_no_grad = f_BCl.detach()
        f_rest = z_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []

        assert self.segment_nums[-1] == L

        num_scales = len(self.segment_nums)

        for scale_idx, segment_num in enumerate(self.segment_nums):
            z_NC = (
                F.interpolate(f_rest, size=segment_num, mode='area').permute(0, 2, 1).reshape(-1, C)
                if (scale_idx != num_scales - 1)
                else f_rest.permute(0, 2, 1).reshape(-1, C)
            )
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.quantize_embedding.weight.data.T, dim=0), dim=1)
            else:
                raise "TODO"

            idx_Bl = idx_N.view(B, segment_num)
            h_BCl = (
                F.interpolate(
                    self.quantize_embedding(idx_Bl).permute(0, 2, 1), size=L, mode='area').contiguous()
                    if (scale_idx != num_scales - 1)
                    else self.quantize_embedding(idx_Bl).permute(0, 2, 1).contiguous()
            )
            h_BCl = self.quant_resi[scale_idx/(num_scales-1)](h_BCl)
            f_hat.add_(h_BCl)
            f_rest.sub_(h_BCl)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, segment_num))

        return f_hat_or_idx_Bl

    def idxBl_to_par_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """
        quantized embedding indices to teacher forcing input
        :param gt_ms_idx_Bl: 不同尺度下的量化嵌入的索引
        :return:
        """
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        L = self.segment_nums[-1]
        num_scales = len(self.segment_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, L, dtype=torch.float32)
        segment_num_next: int = self.segment_nums[0]

        for scale_idx in range(num_scales-1):
            h_BCl = F.interpolate(self.quantize_embedding(gt_ms_idx_Bl[scale_idx]).transpose_(1, 2).view(B, C, segment_num_next), size=L, mode="area")
            f_hat.add_(self.quant_resi[scale_idx/(num_scales-1)](h_BCl))
            segment_num_next = self.segment_nums[scale_idx + 1]
            next_scales.append(F.interpolate(f_hat, size=segment_num_next, mode='area').view(B, C, -1).transpose(1, 2))

        return torch.cat(next_scales, dim=1) if len(next_scales) else None

    def get_next_autoregressive_input(self, scale_idx: int, num_scales: int, f_hat: torch.Tensor, h_BCl: torch.Tensor):
        L = self.segment_nums[-1]
        if scale_idx != num_scales-1:
            h = self.quant_resi[scale_idx / (num_scales - 1)](F.interpolate(h_BCl, size=L, mode="area"))
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=self.segment_nums[scale_idx + 1], mode="area")
        else:
            h = self.quant_resi[scale_idx / (num_scales - 1)](h_BCl)
            f_hat.add_(h)
            return f_hat, f_hat

