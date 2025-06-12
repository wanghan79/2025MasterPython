import torch
import torch.nn as nn
import torch.optim as optim
class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceCNN, self).__init__()

        # 第一个卷积层: 输入通道1，输出通道32，卷积核大小3x3，步幅1，padding=1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 第二个卷积层: 输入通道32，输出通道64，卷积核大小3x3，步幅1，padding=1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 第三个卷积层: 输入通道64，输出通道128，卷积核大小3x3，步幅1，padding=1
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # 全连接层，卷积层输出后展平
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # 假设输入图片大小为64x64
        self.fc2 = nn.Linear(256, num_classes)  # 输出 num_classes 个类别

        # 批归一化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷积 + 激活 + 池化
        x = self.relu(self.bn1(self.conv1(x)))  # 第一个卷积层
        x = nn.MaxPool2d(2)(x)  # 最大池化，池化窗口大小2x2

        x = self.relu(self.bn2(self.conv2(x)))  # 第二个卷积层
        x = nn.MaxPool2d(2)(x)  # 最大池化，池化窗口大小2x2

        x = self.relu(self.bn3(self.conv3(x)))  # 第三个卷积层
        x = nn.MaxPool2d(2)(x)  # 最大池化，池化窗口大小2x2

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # 最后一层是 Softmax 输出，自动由 CrossEntropyLoss 处理
        return x
