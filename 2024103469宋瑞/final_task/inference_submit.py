import torch
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.morphology import label, opening, remove_small_objects
from scipy.ndimage import binary_fill_holes

import config
from model import UNet
from utils import load_checkpoint


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def post_process(mask, min_size=15):
    """
    Post-processing on the predicted mask.
    1. Fills holes in the mask.
    2. Removes small objects.
    """
    # Fill holes
    processed_mask = binary_fill_holes(mask).astype(np.uint8)

    # Remove small objects
    processed_mask = remove_small_objects(processed_mask.astype(bool), min_size=min_size).astype(np.uint8)

    return processed_mask


def main():
    print("Loading model...")
    model = UNet(n_channels=3, n_classes=1).to(config.DEVICE)
    model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth.tar")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    load_checkpoint(torch.load(model_path, map_location=config.DEVICE), model)
    model.eval()

    test_ids = os.listdir(config.TEST_DIR)
    new_test_ids = []
    rles = []

    print("Predicting on test set and generating RLE...")
    for test_id in tqdm(test_ids, total=len(test_ids)):
        path = os.path.join(config.TEST_DIR, test_id, 'images', f'{test_id}.png')

        img = cv2.imread(path)
        original_height, original_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pre-process image like validation set
        augmentations = config.val_transform(image=img)
        img_tensor = augmentations['image'].unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            pred_mask = model(img_tensor)
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = (pred_mask > 0.5).cpu().numpy().squeeze(0).squeeze(0)  # (H, W)

        # Resize mask to original image size
        pred_mask_resized = cv2.resize(pred_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Post-processing
        processed_mask = post_process(pred_mask_resized, min_size=10)

        # Instance Segmentation using watershed or label
        # `label` from skimage is simpler and often sufficient
        labeled_mask, num_labels = label(processed_mask, connectivity=2, return_num=True)

        if num_labels == 0:
            # Handle case with no nuclei found
            new_test_ids.append(test_id)
            rles.append("")
        else:
            for i in range(1, num_labels + 1):
                instance_mask = (labeled_mask == i)
                rle = rle_encode(instance_mask)
                new_test_ids.append(test_id)
                rles.append(rle)

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = rles
    sub.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully.")


if __name__ == "__main__":
    main()