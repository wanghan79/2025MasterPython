# train.py 

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from config import Config
from utils.utils import TextDataset, SimpleTokenizer, setup_logger
from model.transformer_model import TransformerClassifier, count_parameters

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
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.plot_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    setup_logger(os.path.join(cfg.log_dir, "training.log"))

    logging.info("Loading data and building vocab...")
    with open(cfg.train_file, 'r', encoding='utf-8') as f:
        texts = [line.strip().split('\t')[1] for line in f]

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)

    train_dataset = TextDataset(cfg.train_file, tokenizer, cfg.max_len)
    dev_dataset = TextDataset(cfg.dev_file, tokenizer, cfg.max_len)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=cfg.batch_size)

    logging.info("Building model...")
    model = TransformerClassifier(cfg).to(cfg.device)
    total_params, layer_params = count_parameters(model)
    logging.info(f"Total Parameters: {total_params}")
    for name, count in layer_params.items():
        logging.info(f"{name}: {count}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_losses = []
    best_acc = 0.0

    logging.info("Starting training...")
    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{cfg.num_epochs}], Step [{i+1}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        dev_acc, dev_preds, dev_labels = evaluate(model, dev_loader, cfg.device)
        logging.info(f"Epoch [{epoch+1}] completed. Avg Loss: {avg_loss:.4f}, Dev Acc: {dev_acc:.4f}")

        # 生成混淆矩阵
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model.state_dict(), cfg.model_save_path)
            torch.save(model, "model/transformer_model_1.pth")
            logging.info("Best model saved.")
            
            # 绘制并保存混淆矩阵
            plot_confusion_matrix(dev_preds, dev_labels, os.path.join(cfg.plot_dir, "confusion_matrix.png"))
            logging.info("Confusion matrix saved.")

    # Plot loss
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(cfg.plot_dir, "loss_curve.png"))
    logging.info("Training complete. Loss curve saved.")

if __name__ == "__main__":
    import logging
    main()
