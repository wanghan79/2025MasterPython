import time
import torch
import MyNN
import torch.nn as nn
import transforms
from MyDataset import MyDataset
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
writer = SummaryWriter("logs")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用GPU")
else:
    device = torch.device("cpu")
    print("GPU用不了")
device = torch.device("cpu")
transform = transforms.Compose(
    [

        transforms.Resize(224, antialias=True),
        transforms.RandomRotation(10),  # 随机旋转（-10到10度之间）
        transforms.RandomCrop(224),
        transforms.ZeroOneNormalize(),
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.RandomErasing(),  # 随机擦除
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)
transform2 = transforms.Compose(
    [
        transforms.Resize(224, antialias=True),
        transforms.CenterCrop(224),
        transforms.ZeroOneNormalize(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)
data = MyDataset("./data/PetImages", ["0", "1"], transform=transform2, device=device, imgSize=32)
data_loader = DataLoader(dataset=data, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

index = 1
for _ in range(10):
    for i in data_loader:
        inputs, labels = i
        writer.add_images("imgs",inputs,global_step=index)
        index+=1
        print(index)
    writer.close()