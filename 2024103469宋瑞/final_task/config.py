import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "./data/stage1_train/"
TEST_DIR = "./data/stage1_test/"
CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR = "./logs/"
LOAD_MODEL = False
SAVE_MODEL = True

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50 # 增加轮数以获得更好效果
NUM_WORKERS = 4
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
PIN_MEMORY = True

# Data Augmentation
# 训练集增强
train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

# 验证集/测试集只做标准化和尺寸调整
val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)