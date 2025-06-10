import argparse
import torch
import torch.nn as nn
from model import SentimentClassifier
from data_loader import get_dataloader
from sklearn.metrics import classification_report


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, dev_loader = get_dataloader(
        args.train_path, args.dev_path,
        args.bert_model, args.batch_size, args.max_len
    )
    model = SentimentClassifier(args.bert_model, num_labels=2).to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in dev_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    report = classification_report(all_labels, all_preds, digits=4)
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BERT Model")
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--dev_path', type=str, required=True)
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    evaluate(args)
