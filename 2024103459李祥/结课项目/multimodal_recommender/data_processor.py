"""
数据处理主模块
负责数据生成、预处理和加载
"""

import os
import pandas as pd
from utils.data_utils import SyntheticDataGenerator, create_data_loaders
from config import Config


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config=None):
        self.config = config if config else Config()
        self.data_generator = SyntheticDataGenerator(
            num_users=self.config.num_users,
            num_items=self.config.num_items,
            num_interactions=self.config.num_interactions,
            sparsity=self.config.sparsity
        )
    
    def prepare_data(self, force_regenerate=False):
        """准备数据集"""
        data_file = os.path.join(self.config.data_dir, 'interactions.csv')
        
        # 检查数据是否已存在
        if os.path.exists(data_file) and not force_regenerate:
            print("数据集已存在，直接加载...")
            return pd.read_csv(data_file)
        
        # 生成新数据集
        print("生成新的数据集...")
        dataset = self.data_generator.generate_complete_dataset(self.config.data_dir)
        return dataset
    
    def get_data_loaders(self, force_regenerate=False):
        """获取数据加载器"""
        # 准备数据
        self.prepare_data(force_regenerate)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            data_path=os.path.join(self.config.data_dir, 'interactions.csv'),
            batch_size=self.config.batch_size,
            test_ratio=self.config.test_ratio,
            val_ratio=self.config.val_ratio
        )
        
        return train_loader, val_loader, test_loader
    
    def get_data_statistics(self):
        """获取数据统计信息"""
        data_file = os.path.join(self.config.data_dir, 'interactions.csv')
        
        if not os.path.exists(data_file):
            print("数据文件不存在，请先生成数据集")
            return None
        
        df = pd.read_csv(data_file)
        
        stats = {
            'total_interactions': len(df),
            'num_users': df['user_id'].nunique(),
            'num_items': df['item_id'].nunique(),
            'avg_rating': df['rating'].mean(),
            'rating_std': df['rating'].std(),
            'sparsity': 1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique())
        }
        
        return stats
    
    def print_data_info(self):
        """打印数据信息"""
        stats = self.get_data_statistics()
        
        if stats is None:
            return
        
        print("=" * 50)
        print("数据集统计信息")
        print("=" * 50)
        print(f"总交互数: {stats['total_interactions']:,}")
        print(f"用户数量: {stats['num_users']:,}")
        print(f"物品数量: {stats['num_items']:,}")
        print(f"平均评分: {stats['avg_rating']:.2f}")
        print(f"评分标准差: {stats['rating_std']:.2f}")
        print(f"稀疏度: {stats['sparsity']:.2%}")
        print("=" * 50)


if __name__ == "__main__":
    # 测试数据处理器
    processor = DataProcessor()
    
    # 生成数据
    processor.prepare_data(force_regenerate=True)
    
    # 打印数据信息
    processor.print_data_info()
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = processor.get_data_loaders()
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 测试一个批次
    for batch in train_loader:
        print("批次数据形状:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")
        break
