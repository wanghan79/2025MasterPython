import argparse
import sys
import random
import os
import time
import logging
from datetime import datetime
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

# 第三方库导入
from thop import profile
from torchmetrics.functional import structural_similarity_index_measure as tmf_ssim
from torchmetrics.functional import peak_signal_noise_ratio as tmf_psnr

# 自定义模块导入
from utils.scheduler import GradualWarmupScheduler
from model.UNet3d_TMT import DetiltUNet3DS
from model.TMT import TMT_MS
import utils.losses as losses
from utils import utils_image as util
from utils.general import create_log_folder, get_cuda_info, find_latest_checkpoint
from data.dataset_video_train import DataLoaderTurbVideo
from utils.options import dict2str, parse
from utils.dist_util import get_dist_info, init_dist
from utils.logger import MessageLogger, get_dist_info, get_root_logger, init_tb_logger, init_wandb_logger, get_env_info


class VideoDeblurTrainer:
    """视频去模糊训练器，负责模型训练的整体流程"""
    
    def __init__(self, opt):
        """初始化训练器"""
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_frames = opt['num_frames']
        self.gpu_count = torch.cuda.device_count()
        
        # 初始化路径
        self.run_name = opt['name'] + '_' + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        self.run_path = os.path.join(opt['log_path'], self.run_name)
        self.result_img_path, self.path_ckpt, self.path_scipts = self._create_directories()
        
        # 初始化日志
        self.logger = self._init_logger()
        self.logger.info(f"运行路径: {self.run_path}")
        
        # 设置随机种子
        self._set_random_seed()
        
        # 初始化模型
        self.model = self._init_model()
        self.model_tilt = self._init_tilt_model()
        
        # 初始化数据加载器
        self.train_loader, self.val_loader = self._init_data_loaders()
        
        # 初始化优化器和调度器
        self.optimizer, self.scheduler = self._init_optimizer_scheduler()
        
        # 初始化损失函数
        self.criterion_char = losses.CharbonnierLoss()
        self.criterion_edge = losses.EdgeLoss3D()
        
        # 初始化训练状态
        self.best_psnr = 0
        self.best_epoch = 0
        self.iter_count = 1
        self.start_epoch = 1
        
        # 恢复训练状态（如果需要）
        if opt['load']:
            self._resume_training()
            
        # 多GPU支持
        if self.gpu_count > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(self.gpu_count)]).to(self.device)
    
    def _create_directories(self):
        """创建训练所需的目录"""
        if not os.path.exists(self.run_path):
            os.makedirs(self.run_path, exist_ok=True)
        return create_log_folder(self.run_path)
    
    def _init_logger(self):
        """初始化日志记录器"""
        log_file = os.path.join(self.run_path, f"train_{self.opt['name']}_{self._get_time_str()}.log")
        logger = get_root_logger(
            logger_name='basicsr', log_level=logging.INFO, log_file=log_file
        )
        logger.info(get_env_info())
        logger.info(dict2str(self.opt))
        return logger
    
    def _set_random_seed(self):
        """设置随机种子以确保可复现性"""
        seed = self.opt.get('manual_seed')
        if seed is None:
            seed = random.randint(1, 10000)
            self.opt['manual_seed'] = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.logger.info(f"随机种子设置为: {seed}")
    
    def _init_model(self):
        """初始化主模型"""
        model = TMT_MS(
            num_blocks=[2, 3, 3, 4], 
            num_refinement_blocks=2, 
            n_frames=self.n_frames, 
            att_type='shuffle'
        ).to(self.device)
        
        # 打印模型信息
        self._print_model_info(model)
        
        return model
    
    def _print_model_info(self, model):
        """打印模型结构和参数量信息"""
        file_path = f"model_{self.opt['name']}.txt"
        original_stdout = sys.stdout
        
        with open(file_path, 'w') as file:
            sys.stdout = file
            print(model)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"总参数: {total_params/1e6:.2f}M")
            print(f"可训练参数: {trainable_params/1e6:.2f}M")
            
            # 计算FLOPs
            if torch.cuda.is_available():
                image = torch.randn(1, 3, self.n_frames, 120, 120).cuda()
                flops, params = profile(model, (image,))
                print(f"GFLOPS: {flops/1e9:.2f}G --- params: {params}")
        
        sys.stdout = original_stdout
        self.logger.info(f"模型信息已保存到: {file_path}")
    
    def _init_tilt_model(self):
        """初始化倾斜校正模型"""
        model_tilt = DetiltUNet3DS(norm='LN', residual='pool', conv_type='dw').to(self.device)
        
        # 加载预训练权重
        if self.opt['path_tilt']:
            ckpt_tilt = torch.load(self.opt['path_tilt'])
            model_tilt.load_state_dict(ckpt_tilt['state_dict'] if 'state_dict' in ckpt_tilt.keys() else ckpt_tilt)
            self.logger.info(f"已加载倾斜校正模型: {self.opt['path_tilt']}")
        
        # 设置为评估模式
        model_tilt.eval()
        for param in model_tilt.parameters():
            param.requires_grad = False
            
        return model_tilt
    
    def _init_data_loaders(self):
        """初始化训练和验证数据加载器"""
        # 创建训练数据集和加载器
        train_dataset = DataLoaderTurbVideo(
            self.opt['train_path'], 
            num_frames=self.n_frames, 
            patch_size=self.opt['patch_size'],
            noise=0.0001, 
            is_train=True
        )
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.opt['batch_size'], 
            shuffle=True, 
            num_workers=self.opt.get('num_workers', 8),
            drop_last=True, 
            pin_memory=True
        )
        
        # 创建验证数据集和加载器
        val_dataset = DataLoaderTurbVideo(
            self.opt['val_path'], 
            num_frames=self.n_frames, 
            patch_size=self.opt['patch_size'], 
            noise=0.0001,
            is_train=False
        )
        val_loader = DataLoader(
            dataset=val_dataset, 
            batch_size=self.opt['batch_size'], 
            shuffle=False, 
            num_workers=self.opt.get('num_workers', 8),
            drop_last=True, 
            pin_memory=True
        )
        
        self.logger.info(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")
        return train_loader, val_loader
    
    def _init_optimizer_scheduler(self):
        """初始化优化器和学习率调度器"""
        new_lr = self.opt['lr']
        optimizer = optim.Adam(self.model.parameters(), lr=new_lr, betas=(0.9, 0.99), eps=1e-8)
        
        # 学习率调度器
        total_iters = self.opt['iters']
        warmup_iter = 5000
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, total_iters - warmup_iter, eta_min=1e-6
        )
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=warmup_iter, after_scheduler=scheduler_cosine
        )
        
        return optimizer, scheduler
    
    def _resume_training(self):
        """恢复训练状态"""
        if self.opt['load'] == 'latest':
            load_path = find_latest_checkpoint(self.opt['log_path'], self.opt['name'])
            if not load_path:
                self.logger.info(f"搜索最新检查点失败: {self.opt['name']}")
                return
        else:
            load_path = self.opt['load']
        
        self.logger.info(f"从检查点恢复训练: {load_path}")
        checkpoint = torch.load(load_path)
        
        # 加载模型权重
        model_state = checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint
        self.model.load_state_dict(model_state)
        
        # 恢复训练状态
        if not self.opt['start_over']:
            if 'epoch' in checkpoint.keys():
                self.iter_count = checkpoint["epoch"] * len(self.train_loader)
                self.start_epoch = checkpoint["epoch"] + 1
            elif 'iter' in checkpoint.keys():
                self.iter_count = checkpoint["iter"]
            
            # 恢复优化器状态
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                new_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"恢复学习率: {new_lr}")
            
            # 更新调度器状态
            for i in range(1, self.iter_count):
                self.scheduler.step()
    
    def train(self):
        """执行训练过程"""
        # 初始化消息记录器
        tb_logger = init_tb_logger(log_dir=os.path.join('/home/hoo/student-2022/sunwendi/code/tmt/code/tb_logger', self.opt['name']))
        msg_logger = MessageLogger(self.opt, self.iter_count, tb_logger)
        
        self.logger.info(f'''开始训练:
            总迭代数:     {self.opt['iters']}
            起始迭代:     {self.iter_count}
            批次大小:      {self.opt['batch_size']}
            学习率:       {self.optimizer.param_groups[0]['lr']}
            训练样本数:   {len(self.train_loader.dataset)}
            验证样本数:   {len(self.val_loader.dataset)}
            检查点路径:   {self.path_ckpt}
        ''')
        
        # 训练主循环
        total_iters = self.opt['iters']
        epochs = total_iters // len(self.train_loader) + 1
        
        self.model.train()
        
        for epoch in range(self.start_epoch, epochs + 1):
            if self.iter_count > total_iters:
                break
                
            data_time, iter_time = time.time(), time.time()
            current_start_time = time.time()
            current_loss = 0
            train_results = OrderedDict(psnr=[], ssim=[])
            
            # 使用tqdm显示进度条
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch {epoch}')
            
            for i, data in loop:
                data_time = time.time() - data_time
                
                # 零梯度
                self.optimizer.zero_grad()
                
                # 准备数据
                if self.opt['task'] == 'blur':
                    input_ = data[0].to(self.device)
                elif self.opt['task'] == 'turb':
                    input_ = data[1].to(self.device)
                target = data[2].to(self.device)
                
                # 前向传播
                with torch.no_grad():
                    _, _, rectified = self.model_tilt(input_)
                output = self.model(rectified.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
                
                # 计算损失
                if self.iter_count >= 300000:
                    loss = self.criterion_char(output, target) + 0.05 * self.criterion_edge(output, target)
                else:
                    loss = self.criterion_char(output, target)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                current_loss += loss.item()
                self.iter_count += 1
                
                # 评估当前批次
                if self.iter_count % 500 == 0:
                    psnr_batch, ssim_batch = eval_tensor_imgs(
                        target, output, input_, 
                        save_path=self.result_img_path, 
                        kw='train', 
                        iter_count=self.iter_count
                    )
                else:
                    psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_)
                
                train_results['psnr'] += psnr_batch
                train_results['ssim'] += ssim_batch
                
                # 计算平均指标
                avg_psnr = sum(train_results['psnr']) / len(train_results['psnr'])
                avg_ssim = sum(train_results['ssim']) / len(train_results['ssim'])
                
                iter_time = time.time() - iter_time
                
                # 定期保存模型和日志
                if self.iter_count > 1 and self.iter_count % self.opt['print_period'] == 0:
                    # 记录日志
                    log_vars = {
                        'epoch': epoch, 
                        'iter': self.iter_count,
                        'time': iter_time, 
                        'data_time': data_time,
                        'psnr': avg_psnr, 
                        'ssim': avg_ssim,
                        'lr': self.optimizer.param_groups[0]['lr']
                    }
                    msg_logger(log_vars)
                    
                    self.logger.info(
                        f'Training: iters {self.iter_count}/{total_iters} - '
                        f'Time: {time.time()-current_start_time:.6f} - '
                        f'LR: {self.optimizer.param_groups[0]["lr"]:.7f} - '
                        f'Loss: {current_loss/self.opt["print_period"]:8f} - '
                        f'PSNR: {avg_psnr:.2f} dB; SSIM: {avg_ssim:.4f}'
                    )
                    
                    # 保存检查点
                    self._save_checkpoint(f"model_{self.iter_count}.pth")
                    self._save_checkpoint("latest.pth")
                    
                    # 重置统计信息
                    current_start_time = time.time()
                    current_loss = 0
                    train_results = OrderedDict(psnr=[], ssim=[])
                
                # 更新进度条
                loop.set_postfix(
                    iter=self.iter_count, 
                    psnr=f"{avg_psnr:.2f}", 
                    ssim=f"{avg_ssim:.4f}",
                    loss=f"{loss.item():.6f}"
                )
                
                data_time = time.time()
                iter_time = time.time()
                
                # 定期验证
                if self.iter_count > 0 and self.iter_count % self.opt['val_period'] == 0:
                    self._validate(msg_logger)
                    self.model.train()  # 验证后转回训练模式
    
    def _validate(self, msg_logger):
        """执行模型验证"""
        self.logger.info(f"开始验证: Iter {self.iter_count}")
        test_results = OrderedDict(psnr=[], ssim=[])
        eval_loss = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for s, data in enumerate(self.val_loader):
                # 准备数据
                input_ = data[1].to(self.device)
                target = data[2].to(self.device)
                
                # 前向传播
                _, _, rectified = self.model_tilt(input_)
                output = self.model(rectified.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
                
                # 计算损失
                loss = self.criterion_char(output, target)
                eval_loss += loss.item()
                
                # 评估当前批次
                if s % 250 == 0:
                    psnr_batch, ssim_batch = eval_tensor_imgs(
                        target, output, input_, 
                        save_path=self.result_img_path, 
                        kw='val', 
                        iter_count=self.iter_count
                    )
                else:
                    psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_)
                
                test_results['psnr'] += psnr_batch
                test_results['ssim'] += ssim_batch
        
        # 计算平均指标
        avg_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        avg_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        
        # 记录验证结果
        self.logger.info(
            f'Validation: Iters {self.iter_count}/{self.opt["iters"]} - '
            f'Loss: {eval_loss/len(self.val_loader):8f} - '
            f'PSNR: {avg_psnr:.2f} dB; SSIM: {avg_ssim:.4f}'
        )
        
        # 保存最佳模型
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.best_epoch = self.iter_count
            self._save_checkpoint("model_best.pth")
            self.logger.info(f"保存最佳模型: PSNR = {avg_psnr:.2f} dB (之前: {self.best_psnr:.2f} dB)")
    
    def _save_checkpoint(self, filename):
        """保存模型检查点"""
        checkpoint = {
            'iter': self.iter_count,
            'state_dict': self.model.module.state_dict() if self.gpu_count > 1 else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        save_path = os.path.join(self.path_ckpt, filename)
        torch.save(checkpoint, save_path)
        self.logger.info(f"已保存检查点: {save_path}")
    
    def _get_time_str(self):
        """获取当前时间字符串"""
        return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def eval_tensor_imgs(gt, output, input, save_path=None, kw='train', iter_count=0):
    """
    评估张量图像质量
    
    Input images are 5-D in Batch, length, channel, H, W
    output is list of psnr and ssim
    """
    psnr_list = []
    ssim_list = []
    
    for b in range(output.shape[0]):
        for i in range(output.shape[1]):
            # 计算PSNR和SSIM
            img = output[b, i, ...].data.clamp_(0, 1).unsqueeze(0)
            img_gt = gt[b, i, ...].data.clamp_(0, 1).unsqueeze(0)
            psnr_list.append(tmf_psnr(img, img_gt, data_range=1.0).item())
            ssim_list.append(tmf_ssim(img, img_gt, data_range=1.0).item())

            # 保存图像
            if save_path:
                # 准备输入图像
                inp = input[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if inp.ndim == 3:
                    inp = np.transpose(inp, (1, 2, 0))  # CHW-RGB to HWC-BGR
                inp = (inp * 255.0).round().astype(np.uint8)  # float32 to uint8

                # 准备输出图像
                img = output[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if img.ndim == 3:
                    img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
                img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8

                # 准备GT图像
                img_gt = gt[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if img_gt.ndim == 3:
                    img_gt = np.transpose(img_gt, (1, 2, 0))  # CHW-RGB to HWC-BGR
                img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8

                # 拼接并保存图像
                pg_save = Image.fromarray(np.uint8(np.concatenate((inp, img, img_gt), axis=1))).convert('RGB')
                pg_save.save(os.path.join(save_path, f'{kw}_{iter_count}_{b}_{i}.jpg'), "JPEG")
    
    return psnr_list, ssim_list


def parse_options(is_train=True):
    """解析命令行参数和配置文件"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # 分布式设置
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # 随机种子
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def set_random_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    """主函数"""
    # 解析选项
    opt = parse_options(is_train=True)
    
    # 创建并运行训练器
    trainer = VideoDeblurTrainer(opt)
    trainer.train()


if __name__ == '__main__':
    main()    
