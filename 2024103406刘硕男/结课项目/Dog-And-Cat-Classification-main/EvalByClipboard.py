import torch
import MyNN
import torch.nn as nn
import torchvision.transforms as transforms
from MyDataset import MyDataset
from torchvision import models
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import PIL.Image as Image
from PIL import ImageGrab
import clipboard

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用GPU")
else:
    device = torch.device("cpu")
    print("GPU用不了")

# 加载整个模型
model = torch.load('LeNet.pth').to(device)

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)
while True:
    a = input("输入1继续")
    if a=="q":
        break
    else:
        # 从剪切板中获取图片数据

        clipboard_img = clipboard.paste()

        # 如果剪切板中有图片数据
        if clipboard_img is not None:
            # 将图片数据转换为Image对象
            img = ImageGrab.grabclipboard()


            img = transform(img).to(device)
            input_tensor = img.unsqueeze(0)
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                # print(probabilities)
                max_prob_index = torch.argmax(probabilities, dim=1)
                print(max_prob_index.item())