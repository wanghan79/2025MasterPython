import functools
import logging
import time
from typing import Tuple, Optional, Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn

from foldingdiff import losses
from vqvae_modules import Encoder, Decoder, VectorQuantizer


class VQVAEBase(nn.Module):

    angular_loss_fn_dict = {
        "l1": losses.radian_l1_loss,
        "smooth_l1": functools.partial(
            losses.radian_smooth_l1_loss, beta=torch.pi / 10
        ),
    }

    def __init__(self,
                 in_channels: int = 6,  # 输入通道数
                 z_channels: int = 32,  # 潜在空间通道数
                 ch: int = 128,  # 隐藏层通道数
                 dropout_rate: float = 0.0,  # Dropout率

                 # 量化模块参数
                 vocab_size=2048,  # 词汇表大小
                 beta=0.25,  # 损失函数中的β参数
                 using_znorm=True,  # 是否使用Z-Norm
                 quant_conv_ks=3,  # 量化卷积核大小
                 share_quant_resi=4,  # 共享量化残差块数量
                 default_qresi_counts=0,  # 默认量化残差块数量
                 segment_nums=(1, 2, 3, 4, 5, 6, 7, 8)
                 ):
        super().__init__()

        self.vocab_size = vocab_size
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.ch = ch
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.using_znorm = using_znorm
        self.quant_conv_ks = quant_conv_ks
        self.share_quant_resi = share_quant_resi
        self.default_qresi_counts = default_qresi_counts
        self.segment_nums = segment_nums

        self.encoder = Encoder(in_channels=self.in_channels,
                               z_channels=self.z_channels,
                               ch=self.ch,
                               ch_mult=(1, 1, 2, 2, 4),
                               num_res_blocks=2,
                               dropout_rate=self.dropout_rate,
                               using_sa=True,
                               using_mid_sa=True)
        self.pre_quant_conv = nn.Conv1d(self.z_channels,
                                        self.z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.quantizer = VectorQuantizer(vocab_size=self.vocab_size,
                                         z_channels=self.z_channels,
                                         using_znorm=self.using_znorm,
                                         beta=0.25,
                                         default_qresi_counts=self.default_qresi_counts,
                                         segment_nums=self.segment_nums,
                                         quant_resi_ratio=0.5,
                                         share_quant_resi=4)
        self.post_quant_conv = nn.Conv1d(in_channels=self.z_channels,
                                         out_channels=self.z_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
        self.decoder = Decoder(in_channels=self.in_channels,
                               z_channels=self.z_channels,
                               ch=self.ch,
                               ch_mult=(1, 1, 2, 2, 4),
                               num_res_blocks=2,
                               dropout_rate=0.0,
                               using_sa=True,
                               using_mid_sa=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 3, "输入张量的维度应为3" # (batch_size, seq_len, in_channels)
        x = x.permute(0, 2, 1)
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        z_hat, vq_loss = self.quantizer(z)
        z_hat = self.post_quant_conv(z_hat)
        x_recon = self.decoder(z_hat)
        x_recon = x_recon.permute(0, 2, 1)
        return x_recon, vq_loss

    def fhat_to_seq(self, z_hat: torch.Tensor):
        with torch.no_grad():
            return self.decoder(self.post_quant_conv(z_hat)).clamp_(-1, 1)

    def seq_to_idxBl(self, seq_no_grad: torch.Tensor):
        with torch.no_grad():
            f = self.pre_quant_conv(self.encoder(seq_no_grad))
            return self.quantizer.f_to_idxBl_or_fhat(f, to_fhat=False)

    def seq_to_seq(self, seq: torch.Tensor):
        with torch.no_grad():
            seq_recon, _ = self.forward(seq)
            return seq_recon

class VQVAE(VQVAEBase, pl.LightningModule):
    def __init__(
        self,
        loss_key = "smooth_l1",
        lr: float = 5e-5,
        lr_scheduler: Optional[str] = None,
        l2: float = 0.0,
        epochs: int = 1,
        steps_per_epoch: int = 250,  # Dummy value
        **kwargs,
    ):
        VQVAEBase.__init__(self, **kwargs)
        self.learning_rate = lr
        self.lr_scheduler = lr_scheduler
        self.l2_lambda = l2
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.loss = self.angular_loss_fn_dict[loss_key]

        # Epoch counters and timers
        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()

    def _get_loss(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        preds, vq_loss = self.forward(batch["angles"])

        assert preds.ndim == 3, "预测张量的维度应为3" # (batch_size, seq_len, features)

        return self.loss(preds, batch["angles"]), vq_loss

    def training_step(self, batch, batch_idx):
        main_loss, vq_loss = self._get_loss(batch)
        total_loss = main_loss + vq_loss

        self.log("train_loss", total_loss, rank_zero_only=True)
        self.log("main_loss", main_loss, rank_zero_only=True)
        self.log("vq_loss", vq_loss, rank_zero_only=True)

        return total_loss

    def training_epoch_end(self, outputs) -> None:
        """Log average training loss over epoch"""
        losses = torch.stack([o["total_loss"] for o in outputs])
        mean_loss = torch.mean(losses)
        t_delta = time.time() - self.train_epoch_last_time

        logging.info(f"Train loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f} ({t_delta:.2f} seconds)")

        # Increment counter and timers
        self.train_epoch_counter += 1
        self.train_epoch_last_time = time.time()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            main_loss, vq_loss = self._get_loss(batch)
            total_loss = main_loss + vq_loss
            self.log("val_loss", total_loss, rank_zero_only=True)
            self.log("val_main_loss", main_loss, rank_zero_only=True)
            self.log("val_vq_loss", vq_loss, rank_zero_only=True)

        return {"val_loss": total_loss}

    def validation_epoch_end(self, outputs) -> None:
        """Log average validation loss over epoch"""
        losses = torch.stack([o["val_loss"] for o in outputs])
        mean_loss = torch.mean(losses)
        logging.info(f"Validation loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f}")

    def configure_optimizers(self) -> Dict[str, Any]:
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda
        )

        retval = {"optimizer": optim}
        logging.info(f"Using optimizer: {retval}")

        return retval