import argparse
import torch
from transformers import BertTokenizer
from model import SentimentClassifier

LABEL_MAP = {0: 'negative', 1: 'positive'}

def predict(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    model = SentimentClassifier(args.bert_model, num_labels=2)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    text = args.text
    encoding = tokenizer(text, max_length=args.max_len, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        pred = torch.argmax(logits, dim=1).item()
    print(f"Text: {text}\nSentiment: {LABEL_MAP[pred]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-sentence Sentiment Prediction")
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()
    predict(args)
