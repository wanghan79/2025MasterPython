import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import SentimentClassifier
from data_loader import get_dataloader
from utils import setup_logging, save_checkpoint
from tqdm import tqdm

def train(args):
    logger = setup_logging(args.log_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, dev_loader = get_dataloader(
        args.train_path, args.dev_path,
        args.bert_model, args.batch_size, args.max_len
    )
    model = SentimentClassifier(args.bert_model, num_labels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_dev_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in dev_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        dev_acc = correct / total
        logger.info(f"Epoch {epoch}: Dev Acc = {dev_acc:.4f}")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_path = os.path.join(args.output_dir, f"best_model_epoch{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, save_path)
            logger.info(f"Saved best model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT for Sentiment Analysis")
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--dev_path', type=str, required=True)
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_file', type=str, default=None)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
