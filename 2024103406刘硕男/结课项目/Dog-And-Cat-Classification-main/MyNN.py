import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # 3维卷积核，输出6维，输入人3x224x224，输出6x220x220
        x = torch.relu(self.conv1(x))
        # 输入人6x224x224，输出6x110x110
        x = torch.max_pool2d(x, 2)
        # 输入6x110x110，输出16x106x106
        x = torch.relu(self.conv2(x))
        # 丢
        # x = self.dropout(x)

        # 输入16x106x106，输出16x53x53
        x = torch.max_pool2d(x, 2)

        x = x.view(-1, 16 * 53 * 53)
        x = torch.relu(self.fc1(x))
        # 丢
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        # 丢
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


# #(图像尺寸-卷积核尺寸 + 2*填充值)/步长+1
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # 输入 3x224x224 输出 96x55x55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            # 输入 64x55x55 输出96x27x27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 输入 96x27x27 输出 256x27x27
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # 输入 256x27x27 输出 256x13x13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 输入 25613x13 输出 384x13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入 384x13x13 输出 384x13x13
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入384x13x13 输出 256x13x13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入 256x13x13 256x6x6
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNet2(nn.Module):
    def __init__(self, num_classes=1000, drop_out=0.5):
        super(AlexNet2, self).__init__()

        # 残差
        self.downsample = nn.Sequential()

        # 输入 3x224x224 输出 64x55x55
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)

        # 输入 64x55x55 输出64x27x27
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 做一个残差连接
        # 输入 64x27x27 输出 64x27x27
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

        # 增大特征获取
        # 输入 64x27x27 输出 192x27x27
        self.convExpand1 = nn.Conv2d(64, 192, kernel_size=5, padding=2)

        # 缩小图片大小
        # 输入 192x27x27 输出 192x13x13
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 做一个残差连接
        # 输入 192x13x13 输出 192x13x13
        self.conv3 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        # 增大特征获取
        # 输入 192x13x13 输出 384x13x13
        self.convExpand2 = nn.Conv2d(192, 384, kernel_size=3, padding=1)

        # 做一个残差连接
        # 输入 384x13x13 输出 384x13x13
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        # 缩小
        # 输入 384x13x13 输出 256x13x13
        self.convReduce = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        # 做一个残差连接
        # 输入256x13x13 输出 256x13x13
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)

        # 输入 256x13x13 256x6x6
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_out),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_out),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.pool1(x)

        downSample1 = self.downsample(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = x + downSample1

        x = self.convExpand1(x)

        x = self.pool2(x)

        downSample2 = self.downsample(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = x + downSample2

        x = self.convExpand2(x)

        downSample3 = self.downsample(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = x + downSample3

        x = self.convReduce(x)

        downSample4 = self.downsample(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = x + downSample4

        x = self.pool3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class DenseBlockUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlockUnit, self).__init__()
        # self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, grow_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        totalChannels = in_channels
        for i in range(num_layers):
            self.layers.append(DenseBlockUnit(totalChannels, in_channels + grow_rate))
            in_channels += grow_rate
            totalChannels += in_channels
        self.bn = nn.BatchNorm2d(totalChannels)
        self.relu = nn.ReLU(totalChannels)
        self.conv = nn.Conv2d(totalChannels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, 1)))

            # 1 64   features = [64]
            # 2 160  features = [64,96]
            # 3 320  features = [64,96,160]
        x = torch.cat(features, 1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class RenseNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(RenseNet, self).__init__()
        # 输入 3x224x224 输出 64x112x112
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # 数据归一化
        self.bn1 = nn.BatchNorm2d(64)
        # 数据非线性激活
        self.relu = nn.ReLU(inplace=True)
        # 输入 64x112x112 输出 64x56x56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 恒等映射
        self.ds = nn.Sequential()

        # 3个 Dense Block
        self.layers = nn.ModuleList()
        for i in range(3):
            self.layers.append(DenseBlock(64, 64, 16, 3))

        # 缩小特征
        # 输入 64x56x56 输出 128x28x28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)

        # 丢掉一些
        self.dropout1 = nn.Dropout(0.5)

        # 6个 Dense Block

        self.layers2 = nn.ModuleList()
        for i in range(6):
            self.layers2.append(DenseBlock(128, 128, 16, 3))

        # 缩小特征
        # 输入 128x28x28 输出 256x14x14

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)

        # 丢掉一些
        self.dropout2 = nn.Dropout(0.5)

        self.layers3 = nn.ModuleList()
        for i in range(9):
            self.layers3.append(DenseBlock(256, 256, 16, 3))

        # 缩小特征
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1)

        # 丢掉一些
        self.dropout3 = nn.Dropout(0.5)

        # 缩小特征
        self.conv5 = nn.Conv2d(64, 896, kernel_size=14, stride=1)

        self.fc = nn.Linear(896, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for layer in self.layers:
            downSample = self.ds(x)
            x = layer(x)
            x = x + downSample

        x = self.conv3(x)

        x = self.dropout1(x)

        for layer in self.layers2:
            downSample = self.ds(x)
            x = layer(x)
            x = x + downSample

        x = self.conv4(x)

        x = self.dropout2(x)

        for layer in self.layers3:
            downSample = self.ds(x)
            x = layer(x)
            x = x + downSample

        x = self.conv2(x)

        x = self.dropout3(x)

        x = self.conv5(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels,bias=False)
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x).view(x.size(0), -1)
        max_pool = self.max_pool(x).view(x.size(0), -1)
        avg_fc = self.fc(avg_pool)
        max_fc = self.fc(max_pool)
        channel_att = torch.sigmoid(avg_fc+max_fc).view(x.size(0), x.size(1), 1, 1)
        return channel_att * x


class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size//2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding,bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = torch.sigmoid(self.conv(concat))
        return spatial_att * x


class CBAMModule(nn.Module):
    def __init__(self, in_channels):
        super(CBAMModule, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class CBAMAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(CBAMAlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(64),  # Add CBAM module after the first convolutional layer
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(192),  # Add CBAM module after the second convolutional layer
            nn.GroupNorm(24, 192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(256),  # Add CBAM module after the third convolutional layer
            nn.GroupNorm(32, 256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(256, 4096, kernel_size=6, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        # x = torch.flatten(x, 1)
        x = x.view(-1, self.num_classes)

        return x


class LW_CBAMAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(LW_CBAMAlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(64),  # Add CBAM module after the first convolutional layer
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(192),  # Add CBAM module after the second convolutional layer
            nn.GroupNorm(24, 192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(256),  # Add CBAM module after the third convolutional layer
            nn.GroupNorm(32, 256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


import torch
import torch.nn as nn


# 输出96通道
class InceptionModule11X11(nn.Module):
    def __init__(self, in_channels, out1x1=16, red3x3=16, out3x3=32, red5x5=16, out5x5=32, out1x1pool=16):
        super(InceptionModule11X11, self).__init__()

        # 1x1 Convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out1x1, kernel_size=1)

        # 1x1 Convolution followed by two 3x3 Convolution branch (5x5 convolution decomposition)
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, red5x5, kernel_size=1),

            nn.Conv2d(red5x5, red5x5, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(red5x5, red5x5, kernel_size=(3, 1), padding=(1, 0)),

            nn.Conv2d(red5x5, red5x5, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(red5x5, out5x5, kernel_size=(3, 1), padding=(1, 0))

            # nn.Conv2d(red5x5, red5x5, kernel_size=3, padding=1),
            # nn.Conv2d(red5x5, out5x5, kernel_size=3, padding=1)
        )

        # 1x1 Convolution followed by 3x3 Convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, kernel_size=1),
            nn.Conv2d(red3x3, red3x3, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(red3x3, out3x3, kernel_size=(3, 1), padding=(1, 0))

            # nn.Conv2d(red3x3, out3x3, kernel_size=3, padding=1)
        )

        # 3x3 Max Pooling followed by 1x1 Convolution branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out1x1pool, kernel_size=1)
        )

    def forward(self, x):
        # Forward pass through each branch and concatenate the outputs along the depth dimension
        return torch.cat([self.branch1x1(x), self.branch3x3(x), self.branch5x5(x), self.branch_pool(x)], dim=1)


class InceptionModule5x5(nn.Module):
    def __init__(self, in_channels, out1x1=64, red3x3=64, out3x3=128, out1x1pool=64):
        super(InceptionModule5x5, self).__init__()

        # 1x1 Convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out1x1, kernel_size=1)

        # 1x1 Convolution followed by 3x3 Convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, kernel_size=1),
            nn.Conv2d(red3x3, red3x3, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(red3x3, out3x3, kernel_size=(3, 1), padding=(1, 0))

        )

        # 3x3 Max Pooling followed by 1x1 Convolution branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out1x1pool, kernel_size=1)
        )

    def forward(self, x):
        # Forward pass through each branch and concatenate the outputs along the depth dimension
        return torch.cat([self.branch1x1(x), self.branch3x3(x), self.branch_pool(x)], dim=1)


class InceptionA(nn.Module):
    def __init__(self,in_channels,out_channels:list):
        super(InceptionA,self).__init__()
        self.branch1_seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[3], kernel_size=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(),
            nn.Conv2d(out_channels[3], out_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, padding=1)
        )
        self.branch2_seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[3], kernel_size=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(),
            nn.Conv2d(out_channels[3], out_channels[1], kernel_size=3, padding=1)
        )

        self.branch3_seq = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels[2], kernel_size=1)
        )

        self.branch4_seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[3], kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1_seq(x)

        branch2 = self.branch2_seq(x)

        branch3 = self.branch3_seq(x)

        branch4 = self.branch4_seq(x)

        return torch.cat([branch1,branch2,branch3,branch4],dim=1)

class LW_AlexNet(nn.Module):
    def __init__(self, num_classes=1000,dropout=0.5):
        super(LW_AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # 输入 3x224x224 输出 96x55x55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            # 输入16x55x55 输出 64x55x55
            # InceptionModule11X11(64),
            nn.ReLU(inplace=True),
            # 输入 96x55x55 输出96x27x27
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(96),  # Add CBAM module after the second convolutional layer
            nn.BatchNorm2d(96),
            # 输入 96x27x27 输出 256x27x27
            #nn.Conv2d(96, 256, kernel_size=5, padding=2),
            InceptionA(96,[48,80,80,48]),
            # 输入 96x27x27 输出 256x27x27
            #InceptionModule5x5(96),
            nn.ReLU(inplace=True), 
            # 输入 256x27x27 输出 256x13x13
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(256),  # Add CBAM module after the second convolutional layer
            nn.BatchNorm2d(256),
            # 输入 256x13x13 输出 384x13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入 384x13x13 输出 384x13x13
            nn.Conv2d(384, 384, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),
            # 输入384x13x13 输出 256x13x13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # 替换1x3 3x1
            nn.ReLU(inplace=True),
            # 输入 256x13x13 256x6x6
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(256),  # Add CBAM module after the third convolutional layer
            nn.BatchNorm2d(256),
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        #x = self.avgpool(x)
        x = x.contiguous().view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class HW_AlexNet(nn.Module):
    def __init__(self, num_classes=1000,dropout=0.5):
        super(HW_AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # 输入 3x224x224 输出 64x55x55
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # 输入64x55x55 输出 96x55x55
            InceptionModule11X11(64),
            nn.ReLU(inplace=True),
            # 输入 96x55x55 输出96x27x27
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(96),  # Add CBAM module after the second convolutional layer
            nn.BatchNorm2d(96),
            # 输入 96x27x27 输出 128x27x27
            # nn.Conv2d(64, 128, kernel_size=5, padding=2),
            # 输入 96x27x27 输出 256x27x27
            InceptionModule5x5(96),
            nn.ReLU(inplace=True),
            # 输入 256x27x27 输出 256x13x13
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(256),  # Add CBAM module after the second convolutional layer
            nn.BatchNorm2d(256),
            # 输入 256x13x13 输出 384x13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入 384x13x13 输出 256x13x13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),
            # 输入256x13x13 输出 256x13x13
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # 替换1x3 3x1
            nn.ReLU(inplace=True),
            # 输入 256x13x13 256x6x6
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(256),  # Add CBAM module after the third convolutional layer
            nn.BatchNorm2d(256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(256, 4096, kernel_size=6, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        # x = torch.flatten(x, 1)
        x = x.view(-1, self.num_classes)
        return x
#将第二个卷积层换成inceptionA
class AlexNet1(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(AlexNet1, self).__init__()

        self.features = nn.Sequential(
            # 输入 3x224x224 输出 96x55x55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            # 输入 64x55x55 输出96x27x27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 输入 96x27x27 输出 256x27x27
            #nn.Conv2d(96, 256, kernel_size=5, padding=2),
            InceptionA(96, [48, 80, 80, 48]),
            nn.ReLU(inplace=True),
            # 输入 256x27x27 输出 256x13x13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 输入 25613x13 输出 384x13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入 384x13x13 输出 384x13x13
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入384x13x13 输出 256x13x13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入 256x13x13 256x6x6
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
#在每个池化层之后添加归一化
class AlexNet2(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(AlexNet2, self).__init__()

        self.features = nn.Sequential(
            # 输入 3x224x224 输出 96x55x55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            # 输入 64x55x55 输出96x27x27
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            # 输入 96x27x27 输出 256x27x27
            #nn.Conv2d(96, 256, kernel_size=5, padding=2),
            InceptionA(96, [48, 80, 80, 48]),
            nn.ReLU(inplace=True),
            # 输入 256x27x27 输出 256x13x13
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            # 输入 25613x13 输出 384x13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入 384x13x13 输出 384x13x13
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入384x13x13 输出 256x13x13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入 256x13x13 256x6x6
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
#在每个池化层之后增加注意力机制
class AlexNet_CBAM_1(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(AlexNet_CBAM_1, self).__init__()

        self.features = nn.Sequential(
            # 输入 3x224x224 输出 96x55x55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            # 输入 64x55x55 输出96x27x27
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(96),
            nn.BatchNorm2d(96),
            # 输入 96x27x27 输出 256x27x27
            #nn.Conv2d(96, 256, kernel_size=5, padding=2),
            InceptionA(96, [48, 80, 80, 48]),
            nn.ReLU(inplace=True),
            # 输入 256x27x27 输出 256x13x13
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(256),
            nn.BatchNorm2d(256),
            # 输入 25613x13 输出 384x13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入 384x13x13 输出 384x13x13
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入384x13x13 输出 256x13x13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输入 256x13x13 256x6x6
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAMModule(256),
            nn.BatchNorm2d(256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
#在每个卷积层之后增加注意力机制
class AlexNet_CBAM_2(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(AlexNet_CBAM_2, self).__init__()

        self.features = nn.Sequential(
            # 输入 3x224x224 输出 96x55x55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            CBAMModule(96),
            nn.ReLU(),
            # 输入 64x55x55 输出96x27x27
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            # 输入 96x27x27 输出 256x27x27
            #nn.Conv2d(96, 256, kernel_size=5, padding=2),
            InceptionA(96, [48, 80, 80, 48]),
            CBAMModule(256),
            nn.ReLU(inplace=True),
            # 输入 256x27x27 输出 256x13x13
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.BatchNorm2d(256),
            # 输入 25613x13 输出 384x13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            CBAMModule(384),
            nn.ReLU(inplace=True),
            # 输入 384x13x13 输出 384x13x13
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            CBAMModule(384),
            nn.ReLU(inplace=True),
            # 输入384x13x13 输出 256x13x13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            CBAMModule(256),
            nn.ReLU(inplace=True),
            # 输入 256x13x13 256x6x6
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
#在每个激活层后加注意力机制
class AlexNet_CBAM_3(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(AlexNet_CBAM_3, self).__init__()

        self.features = nn.Sequential(
            # 输入 3x224x224 输出 96x55x55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            CBAMModule(96),
            # 输入 64x55x55 输出96x27x27
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            # 输入 96x27x27 输出 256x27x27
            #nn.Conv2d(96, 256, kernel_size=5, padding=2),
            InceptionA(96, [48, 80, 80, 48]),
            nn.ReLU(inplace=True),
            CBAMModule(256),
            # 输入 256x27x27 输出 256x13x13
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.BatchNorm2d(256),
            # 输入 25613x13 输出 384x13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAMModule(384),
            # 输入 384x13x13 输出 384x13x13
            nn.Conv2d(384, 384, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),
            CBAMModule(384),
            # 输入384x13x13 输出 256x13x13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),
            CBAMModule(256),
            # 输入 256x13x13 256x6x6
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

#在每个BatchNormal后面加注意力机制
class AlexNet_CBAM_4(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(AlexNet_CBAM_4, self).__init__()

        self.features = nn.Sequential(
            # 输入 3x224x224 输出 96x55x55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),

            # 输入 64x55x55 输出96x27x27
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            CBAMModule(96),
            # 输入 96x27x27 输出 256x27x27
            #nn.Conv2d(96, 256, kernel_size=5, padding=2),
            InceptionA(96, [48, 80, 80, 48]),
            nn.ReLU(inplace=True),

            # 输入 256x27x27 输出 256x13x13
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.BatchNorm2d(256),
            CBAMModule(256),
            # 输入 25613x13 输出 384x13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 输入 384x13x13 输出 384x13x13
            nn.Conv2d(384, 384, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            # 输入384x13x13 输出 256x13x13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            # 输入 256x13x13 256x6x6
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            CBAMModule(256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
