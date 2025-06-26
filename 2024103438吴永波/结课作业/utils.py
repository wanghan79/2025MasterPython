import os
import random
import numpy as np
import torch
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt

CITYSCAPES_COLORMAP = [
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
    (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,  0)
]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, is_best, save_dir, filename='last_model.pth'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_path)

def load_checkpoint(model, optimizer, path, device='cuda'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint.get('epoch', 0)
    best_score = checkpoint.get('best_score', 0)
    return model, optimizer, epoch, best_score

def mask_to_color(mask):
    mask = mask.cpu().numpy() if torch.is_tensor(mask) else np.array(mask)
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(CITYSCAPES_COLORMAP):
        color_mask[mask == i] = color
    color_mask[mask == 255] = (0, 0, 0)
    return color_mask

def save_mask(mask, path):
    color_mask = mask_to_color(mask)
    Image.fromarray(color_mask).save(path)

def show_mask(mask, title=None):
    color_mask = mask_to_color(mask)
    plt.imshow(color_mask)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def save_image_grid(images, path, nrow=4):
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, scale_each=True)
    ndarr = grid.mul(255).byte().cpu().numpy().transpose(1,2,0)
    Image.fromarray(ndarr).save(path) 