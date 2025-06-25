import os
import torch

# 数据集配置
DATASET = {
    'name': 'CIFAR10',
    'root': './data',
    'num_classes': 10,
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck'),
    'mean': (0.4914, 0.4822, 0.4465),
    'std': (0.2023, 0.1994, 0.2010)
}

# 模型配置
MODEL = {
    'type': 'resnet18',  # 'simple_cnn' or 'resnet18'
    'pretrained': False,
    'save_dir': './checkpoints',
    'save_name': 'best_model.pth'
}

# 训练配置
TRAIN = {
    'batch_size': 128,
    'epochs': 50,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'pin_memory': True,
    'use_amp': True,  # 混合精度训练
    'gradient_clip': 0.1,
    'early_stopping': {
        'patience': 10,
        'min_delta': 0.01
    }
}

# 数据增强配置
AUGMENTATION = {
    'random_crop': True,
    'crop_size': 32,
    'padding': 4,
    'random_horizontal_flip': True,
    'flip_prob': 0.5,
    'random_rotation': 15,
    'color_jitter': True,
    'brightness': 0.1,
    'contrast': 0.1,
    'saturation': 0.1,
    'hue': 0.1
}

# 学习率调度配置
LR_SCHEDULER = {
    'type': 'cosine',  # 'step', 'plateau', 'cosine'
    'step_size': 10,
    'gamma': 0.1,
    'patience': 5,
    'factor': 0.5,
    'min_lr': 1e-6,
    'warmup_epochs': 5
}

# 创建保存目录
if not os.path.exists(MODEL['save_dir']):
    os.makedirs(MODEL['save_dir'])    