#CBAMAlexNet
import time
import torch
import MyNN
import torch.nn as nn
import torchvision.transforms as transforms
from MyDataset import MyDataset
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

writer = SummaryWriter("logs")
print(torch.cuda.is_available())
#torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用GPU")
else:
    device = torch.device("cpu")
    print("GPU用不了")
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.RandomRotation(10),  # 随机旋转（-10到10度之间）
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整亮度、对比度、饱和度和色相
        transforms.RandomResizedCrop(224),  # 随机裁剪和缩放到固定尺寸
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)
#Lenet = MyNN.AlexNet2(num_classes=2,drop_out=0.5).to(device)
Lenet = MyNN.LW_AlexNet(num_classes=2).to(device)
#Lenet = torch.load('最优模型/LW_AlexNet数据增强_epoch15_loss0.11520256585896776_trueRate94.825.pth')
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(Lenet.parameters(), lr=0.01, momentum=0.9)
for parm in Lenet.parameters():
    parm.requires_grad = True
data = MyDataset("./data/PetImages", ["0", "1"], transform=transform, imgSize=-1)
test_data = MyDataset("./data/PetImages", ["0", "1"], transform=transform, isTrain=False, imgSize=-1)
data_loader = DataLoader(dataset=data, batch_size=256, shuffle=True, num_workers=0, drop_last=False)
test_data_loader = DataLoader(dataset=test_data, batch_size=256, shuffle=False, num_workers=0)
def train(epoch):
    Lenet.train()
    lossTotal = 0
    index = 0
    with tqdm(data_loader, desc=f"第{epoch}轮", ) as t:
        for data in t:
            inputs, labels = data
            labels = labels.to(device)
            inputs = inputs.to(device)
            # 清空梯度
            optimizer.zero_grad()
            outputs = Lenet(inputs)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            #梯度裁剪
            torch.nn.utils.clip_grad_norm_(Lenet.parameters(),20)
            # 更新梯度
            optimizer.step()
            # 累积损失
            # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
            # writer.add_images("图片",inputs,global_step=i)
            writer.add_scalar("LW_AlexNet",global_step=(epoch-1)*len(data_loader)+index,scalar_value=loss)
            t.set_postfix(loss=loss.item())
            lossTotal += loss.item()
            index = index+1

    return lossTotal / len(data_loader)


def test(epoch):
    Lenet.eval()
    trueNu = 0
    total_samples = len(test_data_loader.dataset)
    with torch.no_grad():
        for i in test_data_loader:
            imgs, labels = i
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = Lenet(imgs)
            max_prob_index = torch.argmax(outputs, dim=1)
            trueNu += torch.sum(max_prob_index == labels).item()
    return trueNu / len(test_data)

bestTrueRate = 0
for i in range(1, 50):

    loss = train(i)
    trueRate = test(i)
    if trueRate>bestTrueRate:
        torch.save(Lenet, 'LW_AlexNet_epoch{}_loss{}_trueRate{}.pth'.format(i, loss, trueRate * 100))
        bestTrueRate = trueRate
    print("第{}轮,测试集正确率:{}%".format(i, trueRate * 100))
# 保存Letnet模型
# torch.save(Lenet,'LeNet.pth')M
writer.close()
