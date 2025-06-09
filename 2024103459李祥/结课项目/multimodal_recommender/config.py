"""
配置文件 - 多模态推荐系统
包含模型参数、训练参数、数据参数等配置
"""

import torch
import os

class Config:
    """配置类，包含所有超参数和设置"""
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据参数
    num_users = 1000
    num_items = 2000
    num_interactions = 100000
    sparsity = 0.95  # 稀疏度
    
    # 模型参数
    embedding_dim = 64
    hidden_dim = 256
    num_heads = 4
    num_layers = 2
    dropout = 0.1
    
    # 文本特征参数
    text_max_length = 128
    bert_model_name = 'bert-base-uncased'
    text_feature_dim = 768  # BERT输出维度
    
    # 图像特征参数
    image_size = 224
    image_feature_dim = 2048  # ResNet-50输出维度
    
    # 训练参数
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 1e-5
    num_epochs = 20
    patience = 5  # 早停耐心值
    
    # 评估参数
    top_k_list = [5, 10, 20]
    test_ratio = 0.2
    val_ratio = 0.1
    
    # 文件路径
    data_dir = 'data'
    checkpoint_dir = 'checkpoints'
    log_dir = 'logs'
    result_dir = 'results'
    
    # 模型保存
    model_save_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # 日志配置
    log_level = 'INFO'
    
    # 可视化参数
    figure_size = (10, 6)
    dpi = 300
    
    @classmethod
    def get_model_params(cls):
        """获取模型参数字典"""
        return {
            'num_users': cls.num_users,
            'num_items': cls.num_items,
            'embedding_dim': cls.embedding_dim,
            'hidden_dim': cls.hidden_dim,
            'num_heads': cls.num_heads,
            'num_layers': cls.num_layers,
            'dropout': cls.dropout,
            'text_feature_dim': cls.text_feature_dim,
            'image_feature_dim': cls.image_feature_dim,
            'device': cls.device
        }
    
    @classmethod
    def get_training_params(cls):
        """获取训练参数字典"""
        return {
            'batch_size': cls.batch_size,
            'learning_rate': cls.learning_rate,
            'weight_decay': cls.weight_decay,
            'num_epochs': cls.num_epochs,
            'patience': cls.patience,
            'device': cls.device
        }
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 50)
        print("配置信息")
        print("=" * 50)
        print(f"设备: {cls.device}")
        print(f"用户数量: {cls.num_users}")
        print(f"物品数量: {cls.num_items}")
        print(f"交互数量: {cls.num_interactions}")
        print(f"嵌入维度: {cls.embedding_dim}")
        print(f"隐藏维度: {cls.hidden_dim}")
        print(f"注意力头数: {cls.num_heads}")
        print(f"Transformer层数: {cls.num_layers}")
        print(f"批次大小: {cls.batch_size}")
        print(f"学习率: {cls.learning_rate}")
        print(f"训练轮次: {cls.num_epochs}")
        print("=" * 50)
