import os

class Config:
    # 数据集路径
    data_root = './data/Cityscapes'
    train_images = os.path.join(data_root, 'leftImg8bit/train')
    val_images = os.path.join(data_root, 'leftImg8bit/val')
    train_masks = os.path.join(data_root, 'gtFine/train')
    val_masks = os.path.join(data_root, 'gtFine/val')

    # 类别数（Cityscapes 19类+背景）
    num_classes = 20
    ignore_index = 255

    # 训练参数
    epochs = 100
    batch_size = 8
    lr = 1e-3
    weight_decay = 1e-4
    num_workers = 4
    device = 'cuda'  # 'cuda' or 'cpu'

    # 模型保存与日志
    save_dir = './checkpoints'
    log_dir = './logs'
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    last_model_path = os.path.join(save_dir, 'last_model.pth')

    # 输入尺寸
    input_height = 512
    input_width = 1024

    # 随机种子
    seed = 42

    # TensorBoard
    use_tensorboard = True

    # 其他
    print_interval = 10
    val_interval = 1

config = Config() 