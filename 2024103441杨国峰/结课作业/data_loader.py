import os
import json
from typing import List, Tuple
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader


def read_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items

class SentimentDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer: BertTokenizer, max_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        text = item['text']
        label = item['label']
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)


def get_dataloader(
    train_path: str,
    dev_path: str,
    tokenizer_name: str,
    batch_size: int = 32,
    max_len: int = 128
) -> Tuple[DataLoader, DataLoader]:
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    train_data = read_jsonl(train_path)
    dev_data = read_jsonl(dev_path)
    train_ds = SentimentDataset(train_data, tokenizer, max_len)
    dev_ds = SentimentDataset(dev_data, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)
    return train_loader, dev_loader
