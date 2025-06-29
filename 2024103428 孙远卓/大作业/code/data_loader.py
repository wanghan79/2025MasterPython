import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import config
import nltk
import os
from tqdm import tqdm

# 下载必要的NLTK资源
nltk.download(['punkt', 'wordnet', 'stopwords'])


class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, word2idx, max_len):
        self.reviews = reviews
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]

        # 将评论转换为索引序列，并进行填充或截断
        indexed_review = []
        for word in review:
            if word in self.word2idx:
                indexed_review.append(self.word2idx[word])
            else:
                indexed_review.append(self.word2idx['<unk>'])

        # 截断或填充序列
        if len(indexed_review) < self.max_len:
            # 填充
            indexed_review += [self.word2idx['<pad>']] * (self.max_len - len(indexed_review))
        else:
            # 截断
            indexed_review = indexed_review[:self.max_len]

        return torch.tensor(indexed_review, dtype=torch.long), torch.tensor(label, dtype=torch.float)


def clean_text(text):
    """清洗文本数据"""
    # 转换为小写
    text = text.lower()
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除非字母字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    tokens = word_tokenize(text)
    # 移除停用词
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # 移除短词
    tokens = [word for word in tokens if len(word) > 2]
    return tokens


def build_vocab(texts, max_vocab_size):
    """构建词汇表"""
    # 统计词频
    word_counts = Counter()
    for text in tqdm(texts, desc="Building vocabulary"):
        word_counts.update(text)

    # 构建词汇表
    vocab = ['<pad>', '<unk>']  # 添加特殊标记
    # 添加最常见的词
    for word, count in word_counts.most_common(max_vocab_size - len(vocab)):
        vocab.append(word)

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return word2idx, vocab


def load_and_preprocess_data():
    """加载和预处理数据"""
    cfg = config.Config()

    # 读取数据
    print(f"Loading data from {cfg.data_path}...")
    try:
        df = pd.read_csv(cfg.data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {cfg.data_path}")
        print("Please ensure the dataset file exists in the 'data' directory")
        exit(1)

    # 检查数据集格式并重命名列
    if 'Text' in df.columns and 'Score' in df.columns:
        print("Detected Kaggle Amazon Reviews format. Renaming columns...")
        df = df.rename(columns={'Text': 'review', 'Score': 'sentiment'})
    elif 'review' in df.columns and 'sentiment' in df.columns:
        print("Dataset has expected columns.")
    else:
        print("Error: Dataset columns not recognized.")
        print("Expected columns: 'review' and 'sentiment'")
        print(f"Found columns: {list(df.columns)}")
        print("Please ensure your dataset has the correct columns")
        exit(1)

    # 清理文本
    print("Cleaning text data...")
    df['cleaned_text'] = df['review'].apply(clean_text)

    # 构建词汇表
    print("Building vocabulary...")
    word2idx, vocab = build_vocab(df['cleaned_text'], cfg.max_vocab_size)

    # 划分数据集
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df,
        test_size=1 - cfg.train_ratio,
        random_state=42
    )

    # 创建数据集
    train_dataset = ReviewDataset(
        reviews=train_df['cleaned_text'].tolist(),
        labels=train_df['sentiment'].apply(lambda x: 1 if x >= 4 else 0).tolist(),
        word2idx=word2idx,
        max_len=cfg.max_len
    )

    test_dataset = ReviewDataset(
        reviews=test_df['cleaned_text'].tolist(),
        labels=test_df['sentiment'].apply(lambda x: 1 if x >= 4 else 0).tolist(),
        word2idx=word2idx,
        max_len=cfg.max_len
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Vocabulary size: {len(vocab)}")

    return train_loader, test_loader, word2idx, vocab

def save_vocab(word2idx, path):
    """保存词汇表"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(word2idx, path)
    print(f"Vocabulary saved to {path}")


def load_vocab(path):
    """加载词汇表"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    return torch.load(path)