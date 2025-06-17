import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, PreTrainedTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from typing import List, Dict, Tuple, Any


def get_args() -> argparse.Namespace:
 
    parser = argparse.ArgumentParser(
        description="Advanced Sentiment Analysis Training Script."
    )

    parser.add_argument(
        '--data_file', type=str, default='training.csv',
        help='Path to the training CSV file.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='.',
        help='Directory to save checkpoints and best model.'
    )
    parser.add_argument(
        '--checkpoint_name', type=str, default='checkpoint.pt',
        help='Name for the checkpoint file.'
    )
    parser.add_argument(
        '--best_model_name', type=str, default='best_model.pt',
        help='Name for the best model file.'
    )



    parser.add_argument(
        '--bert_model', type=str, default='bert-base-uncased',
        help='Name of the pre-trained BERT model from Hugging Face.'
    )
    parser.add_argument(
        '--max_length', type=int, default=30,
        help='Maximum sequence length for tokenization.'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.2,
        help='Dropout rate for the classification head.'
    )


    parser.add_argument(
        '--epochs', type=int, default=20,
        help='Total number of training epochs.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size for training and evaluation.'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate for the optimizer.'
    )
    parser.add_argument(
        '--val_split_size', type=float, default=0.15,
        help='Proportion of the dataset to use for validation.'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility.'
    )
    


    parser.add_argument(
        '--min_word_count', type=int, default=3,
        help='Minimum number of words a text sample must have.'
    )

    return parser.parse_args()




class Logger:
    """A simple logger to print formatted messages."""
    def __init__(self, log_file="training_log.txt"):
        self.log_file = log_file
        # Clear log file at the start of a new run
        with open(self.log_file, "w") as f:
            f.write(f"Log started at {datetime.now()}\n")

    def header(self, message: str):
        """Prints a formatted header."""
        line = "=" * (len(message) + 4)
        print(f"\n{line}")
        print(f"  {message}")
        print(line)
        self.log(f"\n{line}\n  {message}\n{line}")

    def info(self, message: str):
        """Prints an info message."""
        print(message)
        self.log(message)
    
    def log(self, message: str):
        """Writes a message to the log file."""
        with open(self.log_file, "a") as f:
            f.write(message + "\n")



def set_seed(seed_value: int):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)



def get_device() -> torch.device:
    """Determines and returns the optimal torch device."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")



LABEL_DIC: Dict[str, int] = {'Negative': 0, 'Positive': 1, 'Neutral': 2, 'Irrelevant': 3}
NUM_LABELS: int = len(LABEL_DIC)

def load_and_split_data(file_path: str, min_words: int, val_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads data, preprocesses it, and splits it into training and validation sets.
    """
    df = pd.read_csv(file_path, header=None)
    df.columns = ['id', 'topic', 'sentiment', 'text']
    
    # Preprocessing
    df = df.dropna(subset=['text', 'sentiment'])
    df['word_count'] = df['text'].apply(lambda x: len(str(x).strip().split()))
    df = df[df['word_count'] >= min_words].copy()
    df['label'] = df['sentiment'].apply(lambda x: LABEL_DIC.get(x))
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Splitting
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df['label']
    )
    return train_df, val_df


class SentimentDataset(Dataset):
    """Custom PyTorch Dataset for tokenizing and serving sentiment data."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer: PreTrainedTokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'token_type_ids': inputs['token_type_ids'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }



class SentimentAnalysisModel(nn.Module):
    """BERT-based model with a custom classification head."""
    def __init__(self, bert_model_name: str, num_labels: int, dropout_rate: float):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids) -> torch.Tensor:
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        return self.classifier_head(pooled_output)


def freeze_layers(model: nn.Module):
    """Freezes all layers except the classifier head and BERT's final encoder layer."""
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'classifier_head' in name or 'bert.encoder.layer.11' in name:
            param.requires_grad = True

def summarize_model(model: nn.Module, logger: Logger):
    """Prints a summary of the model's layers and parameters."""
    logger.header("Model Summary")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"{'Layer Name':<40} {'Parameters':<15} {'Trainable':<10}")
    logger.info("-" * 70)
    for name, param in model.named_parameters():
        is_trainable = "✅" if param.requires_grad else "❌"
        logger.info(f"{name:<40} {param.numel():<15,} {is_trainable:<10}")
    
    logger.info("-" * 70)
    logger.info(f"Total Parameters:     {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Frozen Parameters:    {total_params - trainable_params:,}")



def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """
    Runs a single training epoch.
    """
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(
        data_loader, desc=f"Epoch {epoch}/{total_epochs} [Training]", leave=False
    )

    for batch in progress_bar:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        tt_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(ids, mask, tt_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(data_loader)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates the model on a given dataset (e.g., validation set).
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            tt_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(ids, mask, tt_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return {'loss': avg_loss, 'accuracy': accuracy, 'f1_score': f1}



def save_checkpoint(state: Dict, path: str):
    """Saves a training checkpoint."""
    torch.save(state, path)

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str, device: torch.device) -> Tuple[int, float]:
    """Loads a training checkpoint."""
    if not os.path.exists(path):
        return 0, float('inf')
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    return start_epoch, best_val_loss

 
def main(args: argparse.Namespace):
    """
    Main function to orchestrate the entire training and evaluation process.
    """
    logger = Logger()
    logger.header("Sentiment Analysis Training Initializing")
    
 
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    
 
    logger.header("Loading and Preparing Data")
    train_df, val_df = load_and_split_data(args.data_file, args.min_word_count, args.val_split_size, args.seed)
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    train_dataset = SentimentDataset(train_df.text.tolist(), train_df.label.tolist(), tokenizer, args.max_length)
    val_dataset = SentimentDataset(val_df.text.tolist(), val_df.label.tolist(), tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
 
    logger.header("Initializing Model and Optimizer")
    model = SentimentAnalysisModel(args.bert_model, NUM_LABELS, args.dropout).to(device)
    freeze_layers(model)
    summarize_model(model, logger)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
 
    class_counts = train_df.label.value_counts().sort_index().values
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum() * NUM_LABELS
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    logger.info(f"Using CrossEntropyLoss with class weights: {weights.cpu().numpy().round(2)}")

 
    checkpoint_path = os.path.join(args.output_dir, args.checkpoint_name)
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path, device)
    if start_epoch > 0:
        logger.info(f"Resuming training from epoch {start_epoch + 1}. Best validation loss: {best_val_loss:.4f}")
    else:
        logger.info("Starting training from scratch.")
        
 
    logger.header("Starting Main Training Loop")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        val_f1 = val_metrics['f1_score']

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, args.best_model_name)
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"🎉 New best model saved to '{best_model_path}'")
        
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        save_checkpoint(checkpoint_state, checkpoint_path)

    logger.header("Training Complete")
    logger.info(f"Best validation loss achieved: {best_val_loss:.4f}")
    logger.info(f"Best model saved at: {os.path.join(args.output_dir, args.best_model_name)}")


if __name__ == '__main__':
    arguments = get_args()
    
    main(arguments)