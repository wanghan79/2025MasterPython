import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import random

CITYSCAPES_CLASSES = [
    'unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic',
    'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence',
    'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign',
    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan',
    'trailer', 'train', 'motorcycle', 'bicycle'
]

# Cityscapes官方19类+背景映射
CITYSCAPES_ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18
}

class CityscapesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, input_size=(512, 1024), mode='train', augment=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.mode = mode
        self.augment = augment and (mode == 'train')
        self.image_paths = self._get_image_paths()
        self.mask_paths = self._get_mask_paths()
        assert len(self.image_paths) == len(self.mask_paths), 'Image and mask count mismatch!'
        self.transform = self._get_transform()

    def _get_image_paths(self):
        image_paths = []
        for city in os.listdir(self.image_dir):
            city_path = os.path.join(self.image_dir, city)
            for file in os.listdir(city_path):
                if file.endswith('.png'):
                    image_paths.append(os.path.join(city_path, file))
        image_paths.sort()
        return image_paths

    def _get_mask_paths(self):
        mask_paths = []
        for city in os.listdir(self.mask_dir):
            city_path = os.path.join(self.mask_dir, city)
            for file in os.listdir(city_path):
                if file.endswith('_labelIds.png'):
                    mask_paths.append(os.path.join(city_path, file))
        mask_paths.sort()
        return mask_paths

    def _get_transform(self):
        base_transforms = [
            T.Resize(self.input_size, interpolation=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if self.augment:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.input_size, scale=(0.8, 1.0)),
                *base_transforms
            ])
        else:
            return T.Compose(base_transforms)

    def _mask_transform(self, mask):
        mask = np.array(mask, dtype=np.uint8)
        label_mask = np.ones(mask.shape, dtype=np.uint8) * 255
        for k, v in CITYSCAPES_ID_TO_TRAINID.items():
            label_mask[mask == k] = v
        return Image.fromarray(label_mask)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx])
        mask = self._mask_transform(mask)
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        random.seed(seed)
        torch.manual_seed(seed)
        mask = T.Resize(self.input_size, interpolation=Image.NEAREST)(mask)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask


def get_dataloader(image_dir, mask_dir, input_size=(512, 1024), batch_size=8, num_workers=4, mode='train', augment=True):
    dataset = CityscapesDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        input_size=input_size,
        mode=mode,
        augment=augment
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    return loader 