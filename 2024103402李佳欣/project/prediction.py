import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
import numpy as np
import os
import json
from sklearn.metrics import classification_report, precision_recall_fscore_support
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        input = input.float()
        target = target.float()
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = ((1 - pt) ** self.gamma) * BCE_loss
        return focal_loss.mean() if self.size_average else focal_loss.sum()


class BERTAspectOnly(nn.Module):
    def __init__(self, bert_type, num_cat, gamma=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_type, output_hidden_states=True)
        self.ff_cat = nn.Linear(768, num_cat)
        self.gamma = gamma

    def forward(self, labels_cat, **kwargs):
        outputs = self.bert(**kwargs)
        x = outputs.hidden_states[11]
        mask = kwargs['attention_mask']
        se = x * mask.unsqueeze(2)
        den = mask.sum(dim=1).unsqueeze(1)
        se = se.sum(dim=1) / den
        logits_cat = self.ff_cat(se)
        loss = FocalLoss(gamma=self.gamma)(logits_cat, labels_cat)
        return loss, logits_cat


class Trainer:
    def __init__(self, cfg, learning_rate, beta1, beta2, batch_size, gamma):
        self.device = cfg.device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.domain.bert_mapper)
        self.model = BERTAspectOnly(cfg.domain.bert_mapper, len(cfg.domain.aspect_category_mapper), gamma).to(
            self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, betas=(beta1, beta2))
        self.categories = cfg.domain.aspect_category_mapper
        self.aspect_dict = {i: cat for i, cat in enumerate(self.categories)}
        self.inv_aspect_dict = {cat: i for i, cat in enumerate(self.categories)}
        self.threshold = 0.5
        self.epochs = cfg.epochs
        self.batch_size = batch_size
        self.validation_data_size = cfg.validation_data_size
        self.hyper_validation_size = cfg.hyper_validation_size
        self.root_path = cfg.path_mapper

    def load_data(self, file):
        path = os.path.join(self.root_path, file)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        sentences = []
        labels = []
        for item in data.values():
            sentences.append(item['text'])
            label_vec = [0] * len(self.aspect_dict)
            key = 'label' if 'train' in file else 'aspect'
            for asp in item.get(key, []):
                if asp in self.inv_aspect_dict:
                    label_vec[self.inv_aspect_dict[asp]] = 1
            labels.append(label_vec)
        encoded = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=128)
        label_tensor = torch.tensor(labels).float()
        return TensorDataset(label_tensor, encoded['input_ids'], encoded['token_type_ids'],
                             encoded['attention_mask']), sentences, labels

    def find_best_threshold(self, val_dataloader):
        model = self.model
        device = self.device
        model.eval()
        y_true = []
        y_logits = []
        with torch.no_grad():
            for labels_cat, input_ids, token_type_ids, attention_mask in val_dataloader:
                outputs = model(labels_cat.to(device),
                                input_ids=input_ids.to(device),
                                token_type_ids=token_type_ids.to(device),
                                attention_mask=attention_mask.to(device))
                _, logits_cat = outputs
                y_true.append(labels_cat.cpu())
                y_logits.append(logits_cat.cpu())
        y_true = torch.cat(y_true, dim=0).numpy()
        y_logits = torch.cat(y_logits, dim=0).numpy()
        best_th = 0.5
        best_f1 = 0
        for th in np.arange(0.1, 0.9, 0.05):
            preds = (torch.sigmoid(torch.tensor(y_logits)) > th).int().numpy()
            _, _, f1, _ = precision_recall_fscore_support(y_true, preds, average='micro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        print(f"Best threshold: {best_th:.2f}, Micro-F1: {best_f1:.4f}")
        self.threshold = best_th

    def train_model(self, dataset):
        train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - self.validation_data_size,
                                                                       self.validation_data_size])
        train_loader = DataLoader(train_data, batch_size=self.batch_size)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)
        best_loss = float('inf')
        print("\nStart Training...\n")
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for labels_cat, input_ids, token_type_ids, attention_mask in train_loader:
                self.optimizer.zero_grad()
                loss, _ = self.model(
                    labels_cat.to(self.device),
                    input_ids=input_ids.to(self.device),
                    token_type_ids=token_type_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device)
                )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            val_loss = self.evaluate_loss(val_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), '/working/model.pth')
            print(
                f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        self.find_best_threshold(val_loader)

    def evaluate_loss(self, loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for labels_cat, input_ids, token_type_ids, attention_mask in loader:
                loss, _ = self.model(
                    labels_cat.to(self.device),
                    input_ids=input_ids.to(self.device),
                    token_type_ids=token_type_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device)
                )
                losses.append(loss.item())
        return np.mean(losses)

    def evaluate(self, file='test_aspect.json'):
        _, sentences, y_true = self.load_data(file)
        self.model.load_state_dict(torch.load('/working/model.pth'))
        self.model.eval()
        y_pred = []
        predictions = []
        with torch.no_grad():
            for idx, sentence in enumerate(tqdm(sentences)):
                encoded = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt',
                                         max_length=128).to(self.device)
                _, logits_cat = self.model(torch.zeros((1, len(self.aspect_dict))).to(self.device), **encoded)
                probs = torch.sigmoid(logits_cat)
                preds = (probs > self.threshold).int().cpu().numpy().squeeze(0)
                y_pred.append(preds)
                pred_aspects = [self.aspect_dict[i] for i, val in enumerate(preds) if val == 1]
                true_aspects = [self.aspect_dict[i] for i, val in enumerate(y_true[idx]) if val == 1]
                predictions.append({
                    "text": sentence,
                    "predicted_aspects": pred_aspects,
                    "true_aspects": true_aspects
                })
        output_path = os.path.join(self.root_path, "predictions.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        print("\nAspect Multi-Label Classification Report:")
        report = classification_report(y_true, y_pred, target_names=self.categories, digits=4, zero_division=0,
                                       output_dict=True)
        print(classification_report(y_true, y_pred, target_names=self.categories, digits=4, zero_division=0))
        print(f"\n预测结果已保存至: {output_path}")
        metrics = ['precision', 'recall', 'f1-score']
        data = {m: [report[c][m] for c in self.categories] for m in metrics}
        df_metrics = pd.DataFrame(data, index=self.categories)
        plt.figure(figsize=(12, 6))
        df_metrics.plot(kind='bar')
        plt.title('各类别 Precision/Recall/F1-score')
        plt.ylabel('分数')
        plt.xlabel('类别')
        plt.ylim(0, 1)
        plt.xticks(rotation=30)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_path, 'aspect_metrics_bar.png'))
        plt.show()
        pred_counts = np.sum(y_pred, axis=0)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=self.categories, y=pred_counts)
        plt.title('各类别预测数量分布')
        plt.ylabel('预测数量')
        plt.xlabel('类别')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_path, 'aspect_pred_count.png'))
        plt.show()


class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_mapper = "/working"
    epochs = 3
    validation_data_size = 100
    hyper_validation_size = 0.2

    class Domain:
        name = "academic-thesis-comments"
        bert_mapper = "bert-base-chinese"
        aspect_category_mapper = [
            "理论", "能力", "价值",
            "规范", "相关性", "立场"
        ]

    domain = Domain()


if __name__ == '__main__':
    cfg = Config()
    trainer = Trainer(cfg, learning_rate=2e-5, beta1=0.9, beta2=0.999, batch_size=16, gamma=3)
    dataset, _, _ = trainer.load_data('train-labeled.json')
    trainer.train_model(dataset)
    trainer.evaluate('test_aspect.json')
