import os
import numpy as np
from torch.utils.data import Dataset
import cv2


class BowlDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_ids = [name for name in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, name))]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        # 统一处理训练和测试文件夹的不同结构
        img_folder_path = os.path.join(self.data_dir, image_id, 'images')
        image_path = os.path.join(img_folder_path, f'{image_id}.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_dir = os.path.join(self.data_dir, image_id, 'masks')
        # 如果是测试集，则没有masks文件夹
        if os.path.exists(mask_dir):
            mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)
            for mask_file in os.listdir(mask_dir):
                mask_path = os.path.join(mask_dir, mask_file)
                single_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if single_mask is not None:
                    single_mask = np.expand_dims(single_mask, axis=-1)
                    mask = np.maximum(mask, single_mask)

            mask[mask > 0] = 1.0  # Binarize
        else:  # For test set, create an empty mask placeholder
            mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            # ToTensorV2 for mask does not add channel dimension if it's (H, W), we need (1, H, W)
            mask = mask.unsqueeze(0) if mask.ndim == 2 else mask

        # The new structure requires returning image_id for submission file generation
        return image, mask, image_id