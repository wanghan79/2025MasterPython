# config.py

import os

class Config:
    # 模型参数
    vocab_size = 3000
    embedding_dim = 128
    hidden_dim = 256
    num_heads = 4
    num_layers = 2
    num_classes = 2
    dropout = 0.1
    max_len = 128

    # 训练参数
    batch_size = 32
    num_epochs = 20
    learning_rate = 2e-4

    # 路径设置
    data_dir = "./data"
    train_file = os.path.join(data_dir, "train.tsv")
    dev_file = os.path.join(data_dir, "dev.tsv")
    test_file = os.path.join(data_dir, "test.tsv")

    log_dir = "./logs"
    plot_dir = "./plots"
    model_dir = "./model"
    model_save_path = os.path.join(model_dir, "best_model.pth")

    device = "cuda"  # or 'cpu'
