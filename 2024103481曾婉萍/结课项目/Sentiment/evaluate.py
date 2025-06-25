import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

 
BERT_MODEL_NAME = 'bert-base-uncased'
NUM_LABELS = 4
DROPOUT_RATE = 0.2
MAX_LENGTH = 30
BATCH_SIZE = 32
MODEL_PATH = 'best_model.pt'
VALIDATION_FILE = 'validation.csv'
LABEL_DIC = {'Negative': 0, 'Positive': 1, 'Neutral': 2, 'Irrelevant': 3}

 
class SentimentAnalysisModel(nn.Module):
    def __init__(self, output_size, bert_model_name='bert-base-uncased', dropout_rate=0.2):
        super(SentimentAnalysisModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.drop = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, x = self.bert(input_ids, attention_mask, token_type_ids, return_dict=False)
        x = self.relu(self.drop(self.fc1(x)))
        x = self.relu(self.drop(self.fc2(x)))
        x = self.fc3(x)
        return x

 
class EvalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'token_type_ids': inputs['token_type_ids'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

 
test_df = pd.read_csv(VALIDATION_FILE, header=None)
test_df.columns = ['id', 'topic', 'sentiment', 'text']
test_df['label'] = test_df['sentiment'].apply(lambda x: LABEL_DIC.get(x))
test_df = test_df.dropna(subset=['text', 'label'])
test_df['label'] = test_df['label'].astype(int)

 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentAnalysisModel(output_size=NUM_LABELS, dropout_rate=DROPOUT_RATE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

 
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
eval_dataset = EvalDataset(
    texts=test_df.text.tolist(), labels=test_df.label.tolist(), tokenizer=tokenizer, max_len=MAX_LENGTH
)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)

 
all_predictions = []
all_true_labels = []
with torch.no_grad():
    for batch in tqdm(eval_loader, desc="Calculating Predictions"):
        ids, mask, tt_ids = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)
        outputs = model(ids, mask, tt_ids)
        preds = torch.argmax(outputs, dim=1)
        all_predictions.extend(preds.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())
 
print("\n--- Evaluation Results ---")
print(f"Accuracy:  {accuracy_score(all_true_labels, all_predictions):.4f}")
print(f"Precision (Macro): {precision_score(all_true_labels, all_predictions, average='macro'):.4f}")
print(f"Recall (Macro):    {recall_score(all_true_labels, all_predictions, average='macro'):.4f}")
print(f"F1 Score (Macro):  {f1_score(all_true_labels, all_predictions, average='macro'):.4f}")
print("\n--- Classification Report ---")
target_names = list(LABEL_DIC.keys())
print(classification_report(all_true_labels, all_predictions, target_names=target_names))