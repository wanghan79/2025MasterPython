import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from model import get_model
from data_loader import get_data_loaders

def evaluate_model(model_path, blur_dir, sharp_dir, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = get_model(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 获取数据加载器
    _, val_loader = get_data_loaders(blur_dir, sharp_dir, batch_size)
    
    # 评估指标
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for blur_imgs, sharp_imgs in tqdm(val_loader, desc='Evaluating'):
            blur_imgs = blur_imgs.to(device)
            sharp_imgs = sharp_imgs.to(device)
            
            # 生成去模糊图像
            deblurred_imgs = model(blur_imgs)
            
            # 将图像转换回CPU并转换为numpy数组
            deblurred_imgs = deblurred_imgs.cpu().numpy()
            sharp_imgs = sharp_imgs.cpu().numpy()
            
            # 计算每个批次的PSNR和SSIM
            for i in range(deblurred_imgs.shape[0]):
                deblurred = deblurred_imgs[i].transpose(1, 2, 0)
                sharp = sharp_imgs[i].transpose(1, 2, 0)
                
                # 将值范围从[-1, 1]转换到[0, 1]
                deblurred = (deblurred + 1) / 2
                sharp = (sharp + 1) / 2
                
                # 计算PSNR和SSIM
                total_psnr += psnr(sharp, deblurred)
                total_ssim += ssim(sharp, deblurred, multichannel=True)
                num_samples += 1
    
    # 计算平均指标
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    print(f'Average PSNR: {avg_psnr:.2f} dB')
    print(f'Average SSIM: {avg_ssim:.4f}')
    
    return avg_psnr, avg_ssim

if __name__ == '__main__':
    # 设置数据目录和模型路径
    blur_dir = 'data/blur'
    sharp_dir = 'data/sharp'
    model_path = 'checkpoints/model_epoch_100.pth'
    
    # 开始评估
    evaluate_model(model_path, blur_dir, sharp_dir) 