import torch


class Config:
    # 数据路径
    data_path = "data/review.csv"

    # 预处理配置
    max_vocab_size = 20000
    max_len = 200
    train_ratio = 0.8

    # 训练配置
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型配置
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    output_dim = 1  # 二分类
    dropout = 0.3

    # 保存路径
    model_save_path = "model/best_model.pth"
    vocab_save_path = "model/vocab.pth"