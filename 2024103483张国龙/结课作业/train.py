import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from model import get_model
from data_loader import get_data_loaders

def train_model(blur_dir, sharp_dir, num_epochs=100, batch_size=8, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(blur_dir, sharp_dir, batch_size)
    
    # 初始化模型
    model = get_model(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建保存模型的目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # 训练阶段
        for blur_imgs, sharp_imgs in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            blur_imgs = blur_imgs.to(device)
            sharp_imgs = sharp_imgs.to(device)
            
            # 前向传播
            outputs = model(blur_imgs)
            loss = criterion(outputs, sharp_imgs)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for blur_imgs, sharp_imgs in val_loader:
                blur_imgs = blur_imgs.to(device)
                sharp_imgs = sharp_imgs.to(device)
                
                outputs = model(blur_imgs)
                loss = criterion(outputs, sharp_imgs)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    # 设置数据目录
    blur_dir = 'data/blur'
    sharp_dir = 'data/sharp'
    
    # 开始训练
    train_model(blur_dir, sharp_dir) 