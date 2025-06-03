# utils/utils.py

import torch
from torch.utils.data import Dataset
import os
import logging

def setup_logger(log_path=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len):
        self.samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                label, text = line.strip().split('\t')
                token_ids = tokenizer.encode(text, max_len)
                self.samples.append((torch.tensor(token_ids), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class SimpleTokenizer:
    def __init__(self, vocab=None, max_vocab_size=3000):
        self.word2idx = {"[PAD]": 0, "[UNK]": 1}
        self.idx2word = ["[PAD]", "[UNK]"]
        self.max_vocab_size = max_vocab_size
        if vocab:
            self.build_vocab(vocab)

    def build_vocab(self, texts):
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(text.strip())
        for char, _ in counter.most_common(self.max_vocab_size - 2):
            self.word2idx[char] = len(self.word2idx)
            self.idx2word.append(char)

    def encode(self, text, max_len):
        tokens = [self.word2idx.get(c, 1) for c in text]
        tokens = tokens[:max_len]
        tokens += [0] * (max_len - len(tokens))
        return tokens
