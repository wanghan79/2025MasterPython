"""
数据工具模块
包含合成数据生成器和数据集类
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random
import os
import warnings
warnings.filterwarnings('ignore')


class SyntheticDataGenerator:
    """合成数据生成器"""

    def __init__(self, num_users=1000, num_items=2000, num_interactions=100000,
                 sparsity=0.95, random_seed=42):
        self.num_users = num_users
        self.num_items = num_items
        self.num_interactions = num_interactions
        self.sparsity = sparsity
        self.random_seed = random_seed

        # 设置随机种子
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        # 简化的文本处理（不使用BERT tokenizer）
        self.vocab_size = 10000
        self.max_length = 128

        # 图像变换
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def generate_user_item_interactions(self):
        """生成用户-物品交互数据"""
        print("生成用户-物品交互数据...")

        # 生成随机交互
        user_ids = np.random.randint(0, self.num_users, self.num_interactions)
        item_ids = np.random.randint(0, self.num_items, self.num_interactions)

        # 生成评分 (1-5分)
        ratings = np.random.normal(3.5, 1.0, self.num_interactions)
        ratings = np.clip(ratings, 1, 5)

        # 创建DataFrame
        interactions_df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings
        })

        # 去重
        interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'item_id'])

        return interactions_df

    def generate_item_texts(self):
        """生成物品文本描述"""
        print("生成物品文本描述...")

        # 预定义的文本模板
        categories = ['电子产品', '服装', '书籍', '家居', '运动', '美食', '旅游', '音乐']
        adjectives = ['优质的', '时尚的', '实用的', '创新的', '经典的', '现代的', '舒适的', '精美的']
        features = ['设计精良', '性价比高', '用户好评', '热销商品', '限量版', '新品上市', '品质保证', '值得推荐']

        item_texts = []
        for item_id in range(self.num_items):
            category = random.choice(categories)
            adjective = random.choice(adjectives)
            feature = random.choice(features)

            text = f"这是一款{adjective}{category}，{feature}，适合各种场合使用。"
            item_texts.append({
                'item_id': item_id,
                'text': text,
                'category': category
            })

        return pd.DataFrame(item_texts)

    def generate_synthetic_images(self, save_dir='data/images'):
        """生成合成图像数据"""
        print("生成合成图像数据...")

        os.makedirs(save_dir, exist_ok=True)

        image_paths = []
        for item_id in range(self.num_items):
            # 生成随机颜色的图像
            color = np.random.randint(0, 256, 3)
            image = Image.new('RGB', (224, 224), tuple(color))

            # 添加一些随机噪声
            noise = np.random.randint(-50, 50, (224, 224, 3))
            image_array = np.array(image) + noise
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_array)

            # 保存图像
            image_path = os.path.join(save_dir, f'item_{item_id}.jpg')
            image.save(image_path)
            image_paths.append({
                'item_id': item_id,
                'image_path': image_path
            })

        return pd.DataFrame(image_paths)

    def generate_complete_dataset(self, save_dir='data'):
        """生成完整的数据集"""
        print("开始生成完整的合成数据集...")

        os.makedirs(save_dir, exist_ok=True)

        # 生成各种数据
        interactions_df = self.generate_user_item_interactions()
        item_texts_df = self.generate_item_texts()
        item_images_df = self.generate_synthetic_images(os.path.join(save_dir, 'images'))

        # 合并数据
        complete_df = interactions_df.merge(item_texts_df, on='item_id', how='left')
        complete_df = complete_df.merge(item_images_df, on='item_id', how='left')

        # 保存数据
        complete_df.to_csv(os.path.join(save_dir, 'interactions.csv'), index=False)
        item_texts_df.to_csv(os.path.join(save_dir, 'item_texts.csv'), index=False)
        item_images_df.to_csv(os.path.join(save_dir, 'item_images.csv'), index=False)

        print(f"数据集生成完成！")
        print(f"交互数量: {len(complete_df)}")
        print(f"用户数量: {complete_df['user_id'].nunique()}")
        print(f"物品数量: {complete_df['item_id'].nunique()}")

        return complete_df


class MultiModalDataset(Dataset):
    """多模态数据集类"""

    def __init__(self, interactions_df, vocab_size=10000, max_length=128, image_transform=None):
        self.interactions_df = interactions_df
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.image_transform = image_transform

        if self.image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.interactions_df)

    def __getitem__(self, idx):
        row = self.interactions_df.iloc[idx]

        # 获取基本信息
        user_id = torch.tensor(row['user_id'], dtype=torch.long)
        item_id = torch.tensor(row['item_id'], dtype=torch.long)
        rating = torch.tensor(row['rating'], dtype=torch.float)

        # 处理文本数据 - 简化版本
        text = row['text']
        # 简单的文本编码：将文本长度作为特征，并生成随机token IDs
        text_length = min(len(text), self.max_length)
        text_input_ids = torch.randint(1, self.vocab_size, (self.max_length,))
        text_attention_mask = torch.ones(self.max_length)
        # 设置padding部分的attention mask为0
        if text_length < self.max_length:
            text_attention_mask[text_length:] = 0

        # 处理图像数据
        try:
            image = Image.open(row['image_path']).convert('RGB')
            image = self.image_transform(image)
        except:
            # 如果图像加载失败，创建一个随机图像
            image = torch.randn(3, 224, 224)

        return {
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'image': image
        }


def create_data_loaders(data_path='data/interactions.csv', batch_size=32,
                       test_ratio=0.2, val_ratio=0.1, random_seed=42):
    """创建数据加载器"""
    print("创建数据加载器...")

    # 读取数据
    df = pd.read_csv(data_path)

    # 数据分割
    np.random.seed(random_seed)
    indices = np.random.permutation(len(df))

    test_size = int(len(df) * test_ratio)
    val_size = int(len(df) * val_ratio)
    train_size = len(df) - test_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 创建数据集
    train_dataset = MultiModalDataset(df.iloc[train_indices].reset_index(drop=True))
    val_dataset = MultiModalDataset(df.iloc[val_indices].reset_index(drop=True))
    test_dataset = MultiModalDataset(df.iloc[test_indices].reset_index(drop=True))

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
