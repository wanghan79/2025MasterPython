import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            ConvBlock(512, 256),
            ConvBlock(256, 128),
            ConvBlock(128, 64),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 编码
        features = self.encoder(x)
        # 解码
        output = self.decoder(features)
        return output

def get_model(device):
    model = DeblurNet().to(device)
    return model 