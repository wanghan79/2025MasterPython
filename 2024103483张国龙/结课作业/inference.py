import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import get_model
import matplotlib.pyplot as plt

def deblur_image(model_path, image_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = get_model(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载并处理图像
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 进行推理
    with torch.no_grad():
        output = model(input_tensor)
    
    # 将输出转换回图像
    output = output.squeeze(0).cpu()
    output = output * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
             torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    output = torch.clamp(output, 0, 1)
    
    # 转换为PIL图像
    output_image = transforms.ToPILImage()(output)
    
    # 保存结果
    output_image.save(output_path)
    
    # 显示结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Blurred Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(output_image)
    plt.title('Deblurred Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 设置路径
    model_path = 'checkpoints/model_epoch_100.pth'
    image_path = 'test_images/blurred.jpg'
    output_path = 'test_images/deblurred.jpg'
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 执行去模糊
    deblur_image(model_path, image_path, output_path) 