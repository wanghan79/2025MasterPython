# final_inpainting_extended_v2_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter, ImageEnhance
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import glob
from tqdm import tqdm
import time
import cv2
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# ------------------------- 工具函数 -------------------------
def load_image(path, size=(256, 256)):
    """加载并预处理图像"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)

def load_mask(path, size=(256, 256)):
    """加载并预处理掩码，支持自动边缘柔化"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    mask = Image.open(path).convert("L")


    mask = mask.filter(ImageFilter.GaussianBlur(radius=1))
    return transform(mask)
    


def save_tensor_as_image(tensor, path):
    """保存张量为图像"""
    img = tensor.clamp(0, 1).detach().cpu()
    if img.dim() == 4:
        img = img.squeeze(0)
    utils.save_image(img, path)


def create_advanced_mask(size=(256, 256), mask_type="random"):
    """创建高级掩码，支持多种掩码类型"""
    h, w = size
    mask = torch.zeros(size)
    
    
    if mask_type == "random":
        mask_ratio=0.3
        # 随机矩形掩码
        mask = torch.zeros(size)
        # 创建随机矩形掩码
        h, w = size
        mask_h = int(h * mask_ratio)
        mask_w = int(w * mask_ratio)
        start_h = random.randint(0, h - mask_h)
        start_w = random.randint(0, w - mask_w)
        mask[start_h:start_h + mask_h, start_w:start_w + mask_w] = 1.0
        
        
    elif mask_type == "center":
        # 中心掩码
        mask_h = h // 3
        mask_w = w // 3
        start_h = (h - mask_h) // 2
        start_w = (w - mask_w) // 2
        mask[start_h:start_h + mask_h, start_w:start_w + mask_w] = 1.0
        
        
    elif mask_type == "irregular":
        # 不规则掩码
        num_circles = random.randint(3, 8)
        for _ in range(num_circles):
            center_h = random.randint(0, h-1)
            center_w = random.randint(0, w-1)
            radius = random.randint(10, 30)
            
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            circle_mask = ((y - center_h) ** 2 + (x - center_w) ** 2) <= radius ** 2
            mask[circle_mask] = 1.0
            
            
    elif mask_type == "stripe":
        # 条纹掩码
        stripe_width = random.randint(20, 40)
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        if direction == 'horizontal':
            for i in range(0, h, stripe_width * 2):
                mask[i:min(i + stripe_width, h), :] = 1.0
        elif direction == 'vertical':
            for i in range(0, w, stripe_width * 2):
                mask[:, i:min(i + stripe_width, w)] = 1.0
        else:  # diagonal
            for i in range(-h, w, stripe_width * 2):
                for j in range(h):
                    k = i + j
                    if 0 <= k < w and k < i + stripe_width:
                        mask[j, k] = 1.0
    
    return mask.unsqueeze(0)  # 添加通道维度

# ------------------------- U-Net 模型 -------------------------


class DoubleConv(nn.Module):
    """双卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """改进的UNet模型，支持时间嵌入"""
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super().__init__()

        # 时间嵌入网络
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )

        # 下采样层 - 修正通道数
        self.downs = nn.ModuleList([
            DoubleConv(in_channels, 64),      # 输入 -> 64
            DoubleConv(64, 128),              # 64 -> 128
            DoubleConv(128, 256),             # 128 -> 256
            DoubleConv(256, 512),             # 256 -> 512
        ])


        # 上采样层 - 修正输入通道数以匹配跳跃连接
        self.ups = nn.ModuleList([
            DoubleConv(512 + 512, 256),       # 512(上采样) + 512(跳跃) -> 256
            DoubleConv(256 + 256, 128),       # 256(上采样) + 256(跳跃) -> 128
            DoubleConv(128 + 128, 64),        # 128(上采样) + 128(跳跃) -> 64
            DoubleConv(64 + 64, 64),          # 64(上采样) + 64(跳跃) -> 64
        ])

        # 池化和上采样
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.bottleneck = DoubleConv(512, 512)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # 时间嵌入层 - 调整以匹配实际特征通道数
        self.time_convs = nn.ModuleDict({
            'input': nn.Conv2d(time_dim, in_channels, kernel_size=1),
            'down0': nn.Conv2d(time_dim, 64, kernel_size=1),
            'down1': nn.Conv2d(time_dim, 128, kernel_size=1),
            'down2': nn.Conv2d(time_dim, 256, kernel_size=1),
            'down3': nn.Conv2d(time_dim, 512, kernel_size=1),
            'bottleneck': nn.Conv2d(time_dim, 512, kernel_size=1),
            'up0': nn.Conv2d(time_dim, 256, kernel_size=1),
            'up1': nn.Conv2d(time_dim, 128, kernel_size=1),
            'up2': nn.Conv2d(time_dim, 64, kernel_size=1),
            'up3': nn.Conv2d(time_dim, 64, kernel_size=1),
        })

    def sinusoidal_embedding(self, timesteps, dim):
        """正弦位置编码"""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:  # 处理奇数维度
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

    def forward(self, x, t):
        # 时间嵌入
        t_embed = self.sinusoidal_embedding(t, 256)  # [B, 256]
        t_embed = self.time_mlp(t_embed).unsqueeze(2).unsqueeze(3)  # [B, 256, 1, 1]

        # 在输入层添加时间嵌入
        t_input = self.time_convs['input'](t_embed)
        t_input = t_input.expand_as(x)
        x = x + t_input

        # 编码器路径
        enc_features = []
        down_keys = ['down0', 'down1', 'down2', 'down3']
        
        for i, down in enumerate(self.downs):
            x = down(x)
            
            # 添加时间嵌入
            t_conv = self.time_convs[down_keys[i]](t_embed)
            t_conv = t_conv.expand_as(x)
            x = x + t_conv
            
            enc_features.append(x)
            if i < len(self.downs) - 1:  # 最后一层不进行池化
                x = self.pool(x)

        # 瓶颈层
        x = self.bottleneck(x)
        t_bottleneck = self.time_convs['bottleneck'](t_embed)
        t_bottleneck = t_bottleneck.expand_as(x)
        x = x + t_bottleneck

        # 解码器路径
        up_keys = ['up0', 'up1', 'up2', 'up3']
        
        for i, up in enumerate(self.ups):
            x = self.upsample(x)
            
            # 确保尺寸匹配
            enc_feat = enc_features[-(i+1)]
            if x.shape[2:] != enc_feat.shape[2:]:
                x = F.interpolate(x, size=enc_feat.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([x, enc_feat], dim=1)
            x = up(x)
            
            # 添加时间嵌入
            t_up = self.time_convs[up_keys[i]](t_embed)
            t_up = t_up.expand_as(x)
            x = x + t_up

        return self.final(x)

# ------------------------- 扩散模型 -------------------------

class Diffusion:
    """扩散模型实现"""
    def __init__(self, device, T=1000, beta_start=1e-4, beta_end=0.02):
        self.device = device
        self.T = T
        
        # 噪声调度
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # 预计算用于采样的参数
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_betas = torch.sqrt(self.betas)

    def add_noise(self, x0, t):
        """向图像添加噪声"""
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise

    def denoise_step(self, model, xt, t, mask, original_image):
        """单步去噪"""
        if t == 0:
            # 最后一步，直接预测原图
            with torch.no_grad():
                t_tensor = torch.full((xt.size(0),), t, device=self.device, dtype=torch.long)
                pred_noise = model(xt, t_tensor)
                
                # 使用DDPM公式计算x_{t-1}
                sqrt_alpha_bar = self.sqrt_alpha_bars[t]
                sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]
                
                pred_x0 = (xt - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
                
                # 结合原图像和预测结果
                result = original_image * (1 - mask) + pred_x0 * mask
                return result.clamp(0, 1)
        else:
            with torch.no_grad():
                t_tensor = torch.full((xt.size(0),), t, device=self.device, dtype=torch.long)
                pred_noise = model(xt, t_tensor)
                
                # DDPM去噪公式
                alpha = self.alphas[t]
                alpha_bar = self.alpha_bars[t]
                beta = self.betas[t]
                
                # 计算均值
                coef1 = 1 / torch.sqrt(alpha)
                coef2 = beta / torch.sqrt(1 - alpha_bar)
                mean = coef1 * (xt - coef2 * pred_noise)
                
                # 添加噪声（除了最后一步）
                if t > 0:
                    sigma = torch.sqrt(beta)
                    z = torch.randn_like(xt)
                    xt_prev = mean + sigma * z
                else:
                    xt_prev = mean
                
                # 结合原图像（保持已知区域不变）
                xt_prev = original_image * (1 - mask) + xt_prev * mask
                
                return xt_prev

# ------------------------- Inpainting 函数 -------------------------

def inpainting(image_path, mask_path, output_path, steps=50, model_path=None):
    """执行图像修复"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载图像和掩码
    try:
        image = load_image(image_path).unsqueeze(0).to(device)
        mask = load_mask(mask_path).unsqueeze(0).to(device)
        mask = (mask > 0.5).float()
        print(f"图像形状: {image.shape}, 掩码形状: {mask.shape}")
    except Exception as e:
        print(f"加载图像失败: {e}")
        # 创建测试图像和掩码
        print("创建测试图像和掩码...")
        image = torch.rand(1, 3, 256, 256).to(device)
        mask = create_advanced_mask((256, 256),"random").unsqueeze(0).to(device)

    # 初始化模型和扩散过程
    model = UNet().to(device)
    diffusion = Diffusion(device, T=steps)

    # 加载预训练模型（如果有）
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
    else:
        print("使用随机初始化的模型")

    model.eval()

    # 初始化噪声图像
    xt = torch.randn_like(image).to(device)
    
    # 保存初始状态
    save_tensor_as_image(image, "original.png")
    save_tensor_as_image(mask, "mask.png")
    save_tensor_as_image(xt, "initial_noise.png")

    print("开始修复过程...")
    # 扩散逆过程
    for t in tqdm(range(steps - 1, -1, -1), desc="Inpainting"):
        xt = diffusion.denoise_step(model, xt, t, mask, image)
        
        # 保存中间结果
        if t % max(1, steps // 10) == 0:
            save_tensor_as_image(xt, f"step_{t}.png")

    # 保存最终结果
    save_tensor_as_image(xt, output_path)
    print(f"修复完成，结果保存到: {output_path}")

# ------------------------- 训练数据集 -------------------------

class InpaintingDataset(Dataset):
    """图像修复数据集"""
    def __init__(self, image_folder, size=(256, 256)):
        self.image_paths = []
        
        # 收集图像文件
        if os.path.exists(image_folder):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
        
        if not self.image_paths:
            print(f"警告: 在 {image_folder} 中未找到图像文件")
            # 创建虚拟数据用于测试
            self.image_paths = ['dummy'] * 100
            
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            if self.image_paths[idx] == 'dummy':
                # 生成虚拟数据
                img = torch.rand(3, *self.size)
            else:
                img = Image.open(self.image_paths[idx]).convert("RGB")
                img = self.transform(img)
        except Exception:
            # 如果加载失败，使用随机图像
            img = torch.rand(3, *self.size)
        
        # 创建随机掩码
        mask = create_advanced_mask(self.size)
        
        return img, mask

# ------------------------- 训练函数 -------------------------

def train_model(epochs=5, batch_size=4, lr=1e-4):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    
    # 初始化模型和优化器
    model = UNet().to(device)
    diffusion = Diffusion(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.8)
    
    # 创建数据集和数据加载器
    dataset = InpaintingDataset("./train_images")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"数据集大小: {len(dataset)}")
    
    loss_log = []
    
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for img, mask in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            img = img.to(device)
            mask = mask.to(device)
            
            # 随机选择时间步
            t = torch.randint(0, diffusion.T, (img.size(0),), device=device)
            
            # 添加噪声
            xt, noise = diffusion.add_noise(img, t)
            
            # 预测噪声
            pred_noise = model(xt, t)
            
            # 计算损失（只在掩码区域）
            loss = F.mse_loss(pred_noise * mask, noise * mask)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        loss_log.append(avg_loss)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
        
        # 保存模型
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        
        # 可视化损失
        if len(loss_log) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(loss_log)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True)
            plt.savefig("training_loss.png")
            plt.close()


# ------------------------- 主程序 -------------------------

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    
    # 选择运行模式
    mode = "inpaint"  # 可选: "train" 或 "inpaint"
    
    if mode == "train":
        print("开始训练模式...")
        train_model(epochs=3, batch_size=2, lr=1e-4)
    else:
        print("开始推理模式...")
        inpainting(
            image_path="./1.png",        # 输入图像路径
            mask_path="./mask.png",         # 掩码路径
            output_path="./output.png",  # 输出路径
            steps=50,                    # 扩散步数
            model_path="model_epoch_3.pth"  # 预训练模型路径（可选）
        )