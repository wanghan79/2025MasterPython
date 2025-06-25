#evaluate.py

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from config import Config
from utils.utils import TextDataset, SimpleTokenizer
from model.transformer_model import TransformerClassifier

def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(y.cpu().numpy())
            
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    acc = correct / total
    return acc, preds_list, labels_list

def plot_confusion_matrix(preds, labels, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def main():
    cfg = Config()
    print("Loading vocab and tokenizer...")
    with open(cfg.train_file, 'r', encoding='utf-8') as f:
        texts = [line.strip().split('\t')[1] for line in f]

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)

    print("Preparing test set...")
    test_dataset = TextDataset(cfg.test_file, tokenizer, cfg.max_len)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    print("Loading model...")
    model = TransformerClassifier(cfg).to(cfg.device)
    model.load_state_dict(torch.load(cfg.model_save_path))
    
    print("Evaluating...")
    test_acc, test_preds, test_labels = evaluate(model, test_loader, cfg.device)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 绘制并保存混淆矩阵
    plot_confusion_matrix(test_preds, test_labels, 'confusion_matrix_evaluate.png')
    print("Confusion matrix saved as 'confusion_matrix.png'.")

if __name__ == "__main__":
    main()
