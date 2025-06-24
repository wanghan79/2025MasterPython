import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_cifar10
from model import SimpleCNN, ResNet18
from train import train_model
from evaluate import evaluate_model
from visualize import visualize_predictions, plot_training_history

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 配置参数
config = {
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'model_type': 'resnet',  # 可选 'simple' 或 'resnet'
    'data_augmentation': True,
    'save_path': 'cifar10_model.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def main():
    # 加载数据
    train_loader, test_loader, classes = load_cifar10(
        batch_size=config['batch_size'], 
        data_augmentation=config['data_augmentation']
    )
    
    # 初始化模型
    if config['model_type'] == 'simple':
        model = SimpleCNN().to(config['device'])
    else:
        model = ResNet18().to(config['device'])
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5
    )
    
    # 训练模型
    history = train_model(
        model, train_loader, test_loader, criterion, optimizer, 
        scheduler, config['epochs'], config['device']
    )
    
    # 评估模型
    accuracy = evaluate_model(model, test_loader, config['device'])
    print(f'Final Test Accuracy: {accuracy:.2f}%')
    
    # 保存模型
    torch.save(model.state_dict(), config['save_path'])
    print(f'Model saved to {config["save_path"]}')
    
    # 可视化训练历史
    plot_training_history(history)
    
    # 可视化预测结果
    visualize_predictions(model, test_loader, classes, config['device'])

if __name__ == "__main__":
    main()    