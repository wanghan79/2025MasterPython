import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from model import UNet
from dataset import BowlDataset
from utils import load_checkpoint

# Parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1  # Predict one image at a time
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
DATA_DIR = "./data/stage1_train/"  # Use a portion of training data for visualization
MODEL_CHECKPOINT = "./checkpoints/best_model.pth.tar"  # Replace with your best model path


def visualize_predictions(model, loader, device, num_examples=5):
    model.eval()

    fig, axes = plt.subplots(num_examples, 3, figsize=(12, num_examples * 4))
    fig.suptitle("Image | Ground Truth Mask | Predicted Mask", fontsize=16)

    with torch.no_grad():
        for i, sample in enumerate(loader):
            if i >= num_examples:
                break

            image = sample['image'].to(device)
            mask = sample['mask'].to(device)

            # Prediction
            pred_mask = model(image)
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = (pred_mask > 0.5).float()

            # Move to CPU and convert to numpy for plotting
            image_np = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
            mask_np = mask.squeeze(0).cpu().permute(1, 2, 0).numpy()
            pred_mask_np = pred_mask.squeeze(0).cpu().permute(1, 2, 0).numpy()

            # Plotting
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title("Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mask_np.squeeze(), cmap='gray')
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred_mask_np.squeeze(), cmap='gray')
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    # Load model
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT}")
        return

    load_checkpoint(torch.load(MODEL_CHECKPOINT), model)

    # Dataset and DataLoader for visualization
    # We'll use a few images from the training set to see how well it learned
    vis_dataset = BowlDataset(data_dir=DATA_DIR, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    vis_loader = DataLoader(vis_dataset, batch_size=BATCH_SIZE, shuffle=True)

    visualize_predictions(model, vis_loader, DEVICE, num_examples=5)


if __name__ == "__main__":
    main()