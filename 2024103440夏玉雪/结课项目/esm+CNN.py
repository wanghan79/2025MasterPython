import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, auc
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import numpy as np

# 数据加载
class ProteinInteractionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequenceA = row['sequenceA']
        sequenceB = row['sequenceB']
        label = row['class']

        inputsA = self.tokenizer(sequenceA, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length, add_special_tokens=False)
        inputsB = self.tokenizer(sequenceB, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length, add_special_tokens=False)

        return {
            'input_ids_A': inputsA['input_ids'].squeeze(0),
            'attention_mask_A': inputsA['attention_mask'].squeeze(0),
            'input_ids_B': inputsB['input_ids'].squeeze(0),
            'attention_mask_B': inputsB['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

# ESM 嵌入生成
class ProteinSequenceEmbedding(nn.Module):
    def __init__(self, model_path):
        super(ProteinSequenceEmbedding, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        print(f"Embeddings shape: {embeddings.shape}")  # 打印嵌入的形状
        return embeddings

# CNN 模型
class InteractionPredictorCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters=64, kernel_size=3):
        super(InteractionPredictorCNN, self).__init__()
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(num_filters * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seqA_embedding, seqB_embedding):
        # CNN for sequence A
        print(f"seqA_embedding shape: {seqA_embedding.shape}")
        print(f"seqB_embedding shape: {seqB_embedding.shape}")
        seqA_embedding = seqA_embedding.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        seqA_conv = self.relu(self.conv1(seqA_embedding))
        seqA_conv = self.relu(self.conv2(seqA_conv))
        seqA_conv = self.max_pool(seqA_conv)
        seqA_conv = seqA_conv.permute(0, 2, 1)  # (batch_size, seq_len, num_filters)
        seqA_conv = seqA_conv[:, 0, :]  # Take the first position as representative

        # CNN for sequence B
        seqB_embedding = seqB_embedding.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        seqB_conv = self.relu(self.conv1(seqB_embedding))
        seqB_conv = self.relu(self.conv2(seqB_conv))
        seqB_conv = self.max_pool(seqB_conv)
        seqB_conv = seqB_conv.permute(0, 2, 1)  # (batch_size, seq_len, num_filters)
        seqB_conv = seqB_conv[:, 0, :]  # Take the first position as representative

        # Combine features
        combined = torch.cat((seqA_conv, seqB_conv), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x



# 主函数
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, auc
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import numpy as np

# 数据加载
class ProteinInteractionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequenceA = row['sequenceA']
        sequenceB = row['sequenceB']
        label = row['class']

        inputsA = self.tokenizer(sequenceA, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length, add_special_tokens=False)
        inputsB = self.tokenizer(sequenceB, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length, add_special_tokens=False)

        return {
            'input_ids_A': inputsA['input_ids'].squeeze(0),
            'attention_mask_A': inputsA['attention_mask'].squeeze(0),
            'input_ids_B': inputsB['input_ids'].squeeze(0),
            'attention_mask_B': inputsB['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

# ESM 嵌入生成
class ProteinSequenceEmbedding(nn.Module):
    def __init__(self, model_path):
        super(ProteinSequenceEmbedding, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # (batch_size, seq_len, embedding_dim)
        return embeddings

# CNN 模型
class InteractionPredictorCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters=64, kernel_size=3):
        super(InteractionPredictorCNN, self).__init__()
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(num_filters * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seqA_embedding, seqB_embedding):
        # CNN for sequence A
        seqA_embedding = seqA_embedding.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        seqA_conv = self.relu(self.conv1(seqA_embedding))
        seqA_conv = self.relu(self.conv2(seqA_conv))
        seqA_conv = self.max_pool(seqA_conv)
        seqA_conv = seqA_conv.permute(0, 2, 1)  # (batch_size, seq_len, num_filters)
        seqA_conv = seqA_conv[:, 0, :]  # Take the first position as representative

        # CNN for sequence B
        seqB_embedding = seqB_embedding.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        seqB_conv = self.relu(self.conv1(seqB_embedding))
        seqB_conv = self.relu(self.conv2(seqB_conv))
        seqB_conv = self.max_pool(seqB_conv)
        seqB_conv = seqB_conv.permute(0, 2, 1)  # (batch_size, seq_len, num_filters)
        seqB_conv = seqB_conv[:, 0, :]  # Take the first position as representative

        # Combine features
        combined = torch.cat((seqA_conv, seqB_conv), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 主函数
def main():
    # 数据路径
    train_file = '/mnt/Data6/hjy/STEP/data/train1500.txt'  # 替换为你的训练数据文件路径
    test_file = '/mnt/Data6/hjy/STEP/data/test1000.txt'    # 替换为你的测试数据文件路径

    # 加载数据
    train_data = pd.read_csv(train_file, sep="\t")
    test_data = pd.read_csv(test_file, sep="\t")
    print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")

    # 初始化 ESM 模型
    model_path = "/mnt/Data6/hjy/STEP/esm2/esm2_t12_35M_UR50D"
    protein_seq_tokenizer = ProteinSequenceEmbedding(model_path)

    # 数据集
    train_dataset = ProteinInteractionDataset(train_data, protein_seq_tokenizer.tokenizer)
    test_dataset = ProteinInteractionDataset(test_data, protein_seq_tokenizer.tokenizer)

    # 分层五折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []

    # 创建保存模型的文件夹
    save_dir = "/mnt/Data6/hjy/STEP/new/saved_models"
    os.makedirs(save_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, train_data['class'])):
        print(f"\nTraining fold {fold + 1}")
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # 选择模型：CNN
        model = InteractionPredictorCNN(embedding_dim=480)  # 使用 CNN
        model.to(protein_seq_tokenizer.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 早停机制参数
        early_stopping_patience = 5
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(100):  # 训练 100 个 epoch
            model.train()
            train_loss = 0
            for batch in train_loader:
                input_ids_A = batch['input_ids_A'].to(protein_seq_tokenizer.device)
                attention_mask_A = batch['attention_mask_A'].to(protein_seq_tokenizer.device)
                input_ids_B = batch['input_ids_B'].to(protein_seq_tokenizer.device)
                attention_mask_B = batch['attention_mask_B'].to(protein_seq_tokenizer.device)
                labels = batch['label'].to(protein_seq_tokenizer.device)

                embeddings_A = protein_seq_tokenizer(input_ids_A, attention_mask_A)
                embeddings_B = protein_seq_tokenizer(input_ids_B, attention_mask_B)

                optimizer.zero_grad()
                outputs = model(embeddings_A, embeddings_B)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证集评估
            model.eval()
            val_loss = 0
            val_predictions = []
            val_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids_A = batch['input_ids_A'].to(protein_seq_tokenizer.device)
                    attention_mask_A = batch['attention_mask_A'].to(protein_seq_tokenizer.device)
                    input_ids_B = batch['input_ids_B'].to(protein_seq_tokenizer.device)
                    attention_mask_B = batch['attention_mask_B'].to(protein_seq_tokenizer.device)
                    labels = batch['label'].to(protein_seq_tokenizer.device)

                    embeddings_A = protein_seq_tokenizer(input_ids_A, attention_mask_A)
                    embeddings_B = protein_seq_tokenizer(input_ids_B, attention_mask_B)

                    outputs = model(embeddings_A, embeddings_B)
                    val_loss += criterion(outputs.squeeze(), labels).item()
                    val_predictions.extend(outputs.squeeze().cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            train_loss /= len(train_loader)
          
            print(f"Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        # 计算验证集的评价指标
        val_predictions_binary = (np.array(val_predictions) > 0.5).astype(int)
        val_accuracy = accuracy_score(val_labels, val_predictions_binary)
        val_f1 = f1_score(val_labels, val_predictions_binary)
        val_auc = roc_auc_score(val_labels, val_predictions)
        val_precision = precision_score(val_labels, val_predictions_binary)
        val_recall = recall_score(val_labels, val_predictions_binary)
        val_mcc = matthews_corrcoef(val_labels, val_predictions_binary)
        precision, recall, _ = precision_recall_curve(val_labels, val_predictions)
        val_auc_pr = auc(recall, precision)

        print(f"\nFold {fold + 1} Validation Metrics:")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        print(f"Validation AUC: {val_auc:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall: {val_recall:.4f}")
        print(f"Validation MCC: {val_mcc:.4f}")
        print(f"Validation AUC-PR: {val_auc_pr:.4f}")       

        # 保存模型
        model_path = os.path.join(save_dir, f"model_fold_{fold + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        models.append(model)

    # 测试集预测
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    all_predictions = []

    for model in models:
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids_A = batch['input_ids_A'].to(protein_seq_tokenizer.device)
                attention_mask_A = batch['attention_mask_A'].to(protein_seq_tokenizer.device)
                input_ids_B = batch['input_ids_B'].to(protein_seq_tokenizer.device)
                attention_mask_B = batch['attention_mask_B'].to(protein_seq_tokenizer.device)

                embeddings_A = protein_seq_tokenizer(input_ids_A, attention_mask_A)
                embeddings_B = protein_seq_tokenizer(input_ids_B, attention_mask_B)

                outputs = model(embeddings_A, embeddings_B)
                predictions.extend(outputs.squeeze().cpu().numpy())
        all_predictions.append(predictions)

    # 投票机制
    final_predictions = np.mean(all_predictions, axis=0)
    final_predictions_binary = (final_predictions > 0.5).astype(int)

    # 评估
    test_labels = test_data['class'].values
    accuracy = accuracy_score(test_labels, final_predictions_binary)
    f1 = f1_score(test_labels, final_predictions_binary)
    test_auc = roc_auc_score(test_labels, final_predictions)
    precision = precision_score(test_labels, final_predictions_binary)
    recall = recall_score(test_labels, final_predictions_binary)
    mcc = matthews_corrcoef(test_labels, final_predictions_binary)

    # 计算 AUC-PR
    precision, recall, _ = precision_recall_curve(test_labels, final_predictions)
    test_auc_pr = auc(recall, precision)

    print(f"\nTest Metrics:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test MCC: {mcc:.4f}")
    print(f"Test AUC-PR: {test_auc_pr:.4f}")
def test():
    # 初始化配置
    test_file = '/mnt/Data6/hjy/STEP/data/test1000.txt'
    model_dir = "/mnt/Data6/hjy/STEP/new/saved_models"
    esm_model_path = "/mnt/Data6/hjy/STEP/esm2/esm_2_model"

    # 加载测试数据
    test_data = pd.read_csv(test_file, sep="\t")
    print(f"Test data shape: {test_data.shape}")

    # 初始化ESM模型
    protein_embedder = ProteinSequenceEmbedding(esm_model_path)
    test_dataset = ProteinInteractionDataset(test_data, protein_embedder.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 加载所有保存的模型
    models = []
    for fold in range(1, 6):
        model_path = os.path.join(model_dir, f"model_fold_{fold}.pth")
        model = InteractionPredictorCNN(embedding_dim=480)
        model.load_state_dict(torch.load(model_path))
        model.to(protein_embedder.device)
        model.eval()
        models.append(model)
        print(f"Loaded model from {model_path}")

    # 测试集预测
    all_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(protein_embedder.device) for k, v in batch.items()}
            embeddings_A = protein_embedder(batch['input_ids_A'], batch['attention_mask_A'])
            embeddings_B = protein_embedder(batch['input_ids_B'], batch['attention_mask_B'])

            fold_preds = []
            for model in models:
                outputs = model(embeddings_A, embeddings_B)
                fold_preds.append(outputs.squeeze().cpu().numpy())
            all_predictions.append(np.mean(fold_preds, axis=0))

    final_predictions = np.concatenate(all_predictions)
    final_predictions_binary = (final_predictions > 0.5).astype(int)

    # 评估指标计算（修复变量名冲突）
    test_labels = test_data['class'].values
    accuracy = accuracy_score(test_labels, final_predictions_binary)
    f1 = f1_score(test_labels, final_predictions_binary)
    test_auc = roc_auc_score(test_labels, final_predictions)
    test_precision = precision_score(test_labels, final_predictions_binary)
    test_recall = recall_score(test_labels, final_predictions_binary)
    mcc = matthews_corrcoef(test_labels, final_predictions_binary)
    
    # 计算AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(test_labels, final_predictions)
    test_auc_pr = auc(recall_curve, precision_curve)

    print("\nFinal Test Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUC-PR: {test_auc_pr:.4f}")



if __name__ == "__main__":
    test()