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

# 交叉注意力机制
class CrossAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        query = self.query(query)  # (batch_size, seq_len, embedding_dim)
        key = self.key(key)  # (batch_size, seq_len, embedding_dim)
        value = self.value(value)  # (batch_size, seq_len, embedding_dim)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        attention_scores = attention_scores / np.sqrt(query.size(-1))
        attention_weights = self.softmax(attention_scores)  # (batch_size, seq_len, seq_len)
        context = torch.matmul(attention_weights, value)  # (batch_size, seq_len, embedding_dim)
        return context

# 交叉注意力模型
class InteractionPredictorCrossAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(InteractionPredictorCrossAttention, self).__init__()
        self.cross_attention = CrossAttention(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, seqA_embedding, seqB_embedding):
        # Apply cross-attention
        seqA_context = self.cross_attention(seqA_embedding, seqB_embedding, seqB_embedding)
        seqB_context = self.cross_attention(seqB_embedding, seqA_embedding, seqA_embedding)

        # Take the first position as representative
        seqA_context = seqA_context[:, 0, :]
        seqB_context = seqB_context[:, 0, :]

        # Combine features
        combined = torch.cat((seqA_context, seqB_context), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 主函数
def main():
    # 数据路径
    train_file = '/mnt/Data6/hjy/STEP/data/train1500.txt'  
    test_file = '/mnt/Data6/hjy/STEP/data/test1000.txt'    

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
    save_dir = "/mnt/Data6/hjy/STEP/new/saved_models_cross_attention"
    os.makedirs(save_dir, exist_ok=True)

   


    for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, train_data['class'])):
        print(f"\nTraining fold {fold + 1}")
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # 选择模型：交叉注意力
        model = InteractionPredictorCrossAttention(embedding_dim=480)  # 使用交叉注意力
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
    precision = precision_score(test_labels, final_predictions_binary, average='binary')
    recall = recall_score(test_labels, final_predictions_binary, average='binary')
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
def load_models(device):
    """加载所有5个交叉验证模型"""
    model_dir = "/mnt/Data6/hjy/STEP/new/saved_models_cross_attention"
    models = []
    for fold in range(1, 6):
        model = InteractionPredictorCrossAttention(embedding_dim=480)
        model_path = os.path.join(model_dir, f"model_fold_{fold}.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    return models
def test():
    # 初始化配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载测试数据
    test_file = '/mnt/Data6/hjy/STEP/data/test1000.txt'
    test_data = pd.read_csv(test_file, sep="\t")
    
    # 初始化ESM模型
    esm_model = ProteinSequenceEmbedding("/mnt/Data6/hjy/STEP/esm2/esm2_t12_35M_UR50D")
    
    # 创建测试数据集
    test_dataset = ProteinInteractionDataset(test_data, esm_model.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 加载训练好的模型
    models = load_models(device)
    
    # 进行预测
    all_predictions = []
    with torch.no_grad():
        for model in models:
            model_predictions = []
            for batch in test_loader:
                # 数据转移到设备
                input_ids_A = batch['input_ids_A'].to(device)
                attention_mask_A = batch['attention_mask_A'].to(device)
                input_ids_B = batch['input_ids_B'].to(device)
                attention_mask_B = batch['attention_mask_B'].to(device)
                
                # 生成嵌入
                emb_A = esm_model(input_ids_A, attention_mask_A)
                emb_B = esm_model(input_ids_B, attention_mask_B)
                
                # 预测
                outputs = model(emb_A, emb_B)
                model_predictions.extend(outputs.squeeze().cpu().numpy())
            all_predictions.append(model_predictions)
    
    # 集成预测结果
    final_predictions = np.mean(all_predictions, axis=0)
    final_binary = (final_predictions > 0.5).astype(int)
    
    # 计算指标
    labels = test_data['class'].values
    metrics = {
        "Accuracy": accuracy_score(labels, final_binary),
        "F1": f1_score(labels, final_binary),
        "AUC": roc_auc_score(labels, final_predictions),
        "Precision": precision_score(labels, final_binary),
        "Recall": recall_score(labels, final_binary),
        "MCC": matthews_corrcoef(labels, final_binary)
    }
    
    # 计算AUC-PR
    precisions, recalls, _ = precision_recall_curve(labels, final_predictions)
    metrics["AUC-PR"] = auc(recalls, precisions)
    
    # 打印结果
    print("\nTest Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    test()