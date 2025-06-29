import torch
import torch.nn as nn
import torch.optim as optim
from model import SentimentLSTM, initialize_model, count_parameters
from data_loader import load_and_preprocess_data, save_vocab
import config
import os
from tqdm import tqdm
import numpy as np
from visualize import plot_training_history
from sklearn.metrics import accuracy_score, f1_score


def train_model():
    """训练模型主函数"""
    cfg = config.Config()

    # 创建模型保存目录
    os.makedirs(os.path.dirname(cfg.model_save_path), exist_ok=True)

    # 加载数据
    train_loader, test_loader, word2idx, vocab = load_and_preprocess_data()

    # 保存词汇表
    save_vocab(word2idx, cfg.vocab_save_path)

    # 初始化模型
    model = initialize_model(len(vocab))
    model = model.to(cfg.device)

    # 打印模型信息
    print(f"Model initialized on {cfg.device}")
    print(f"Model has {count_parameters(model):,} trainable parameters")

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # 训练历史记录
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'train_f1': [],
        'test_f1': []
    }

    best_accuracy = 0.0

    # 训练循环
    for epoch in range(cfg.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs} - Training"):
            inputs, labels = batch
            inputs = inputs.to(cfg.device)
            labels = labels.to(cfg.device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs).squeeze(1)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 记录损失
            train_loss += loss.item()

            # 计算预测结果
            preds = torch.round(torch.sigmoid(outputs))
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())

        # 计算训练指标
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)

        # 测试阶段
        test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, criterion, cfg.device)

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_f1'].append(train_f1)
        history['test_f1'].append(test_f1)

        # 打印结果
        print(f"Epoch {epoch + 1}/{cfg.num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), cfg.model_save_path)
            print(f"Saved best model with accuracy: {best_accuracy:.4f}")

    # 绘制训练历史
    plot_training_history(history)

    print("Training completed!")
    return model, history


def evaluate_model(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            preds = torch.round(torch.sigmoid(outputs))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(data_loader)
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds)

    return avg_test_loss, test_acc, test_f1