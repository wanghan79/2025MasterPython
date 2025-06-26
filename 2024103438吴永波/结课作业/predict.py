import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from config import config
from model import UNet
from utils import mask_to_color, show_mask, save_mask
import numpy as np

def load_image(image_path, input_size):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu()
    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.py', help='Config file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save results')
    parser.add_argument('--show', action='store_true', help='Show result mask')
    args = parser.parse_args()

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=config.num_classes).to(device)
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f'Loaded model from {args.model_path}')
    else:
        checkpoint = torch.load(config.best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f'Loaded model from {config.best_model_path}')
    os.makedirs(args.output_dir, exist_ok=True)
    input_size = (config.input_height, config.input_width)

    if os.path.isdir(args.image_path):
        image_files = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path) if f.endswith('.png') or f.endswith('.jpg')]
    else:
        image_files = [args.image_path]

    for image_file in image_files:
        image_tensor = load_image(image_file, input_size)
        pred_mask = predict(model, image_tensor, device)
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        save_path = os.path.join(args.output_dir, base_name + '_mask.png')
        save_mask(pred_mask, save_path)
        print(f'Saved mask to {save_path}')
        if args.show:
            show_mask(pred_mask, title=base_name)

if __name__ == '__main__':
    main() 