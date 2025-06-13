import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

import config
from dataset import BowlDataset
from model import UNet
from utils import dice_score, iou_score, save_checkpoint, load_checkpoint


def train_fn(loader, model, optimizer, loss_fn, scaler, writer, epoch):
    loop = tqdm(loader, leave=True)
    mean_loss, mean_dice, mean_iou = 0, 0, 0

    for batch_idx, (data, targets, _) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.to(device=config.DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        dice = dice_score(predictions, targets)
        iou = iou_score(predictions, targets)
        mean_loss += loss.item()
        mean_dice += dice
        mean_iou += iou

        loop.set_postfix(loss=loss.item(), dice=dice, iou=iou)

    mean_loss /= len(loader)
    mean_dice /= len(loader)
    mean_iou /= len(loader)
    writer.add_scalar("Train/Loss", mean_loss, epoch)
    writer.add_scalar("Train/Dice", mean_dice, epoch)
    writer.add_scalar("Train/IoU", mean_iou, epoch)
    print(f"Epoch {epoch} Train: Loss={mean_loss:.4f}, Dice={mean_dice:.4f}, IoU={mean_iou:.4f}")


def check_accuracy(loader, model, loss_fn, device, writer, epoch):
    print("Checking accuracy on validation set...")
    num_correct, num_pixels = 0, 0
    dice, iou, val_loss = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            val_loss += loss_fn(preds, y).item()
            dice += dice_score(preds, y)
            iou += iou_score(preds, y)

            preds_binary = (torch.sigmoid(preds) > 0.5).float()
            num_correct += (preds_binary == y).sum()
            num_pixels += torch.numel(preds_binary)

    mean_loss = val_loss / len(loader)
    mean_dice = dice / len(loader)
    mean_iou = iou / len(loader)
    accuracy = num_correct / num_pixels * 100

    writer.add_scalar("Val/Loss", mean_loss, epoch)
    writer.add_scalar("Val/Dice", mean_dice, epoch)
    writer.add_scalar("Val/IoU", mean_iou, epoch)
    writer.add_scalar("Val/PixelAccuracy", accuracy, epoch)
    print(f"Val: Loss={mean_loss:.4f}, Dice={mean_dice:.4f}, IoU={mean_iou:.4f}, Acc={accuracy:.2f}%")

    model.train()
    return mean_iou


def main():
    if not os.path.exists(config.CHECKPOINT_DIR): os.makedirs(config.CHECKPOINT_DIR)
    if not os.path.exists(config.LOG_DIR): os.makedirs(config.LOG_DIR)

    writer = SummaryWriter(config.LOG_DIR)

    model = UNet(n_channels=3, n_classes=1).to(config.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.1, verbose=True)

    if config.LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    full_dataset = BowlDataset(data_dir=config.TRAIN_DIR, transform=config.train_transform)

    val_percent = 0.15
    n_val = int(len(full_dataset) * val_percent)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    # Ensure validation set uses validation transforms
    val_dataset.dataset = BowlDataset(data_dir=config.TRAIN_DIR, transform=config.val_transform)
    val_dataset.indices = val_dataset.indices  # This is a bit of a hack, but ensures transform is correct for the subset

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, shuffle=False
    )

    scaler = torch.cuda.amp.GradScaler()
    best_val_iou = -1.0

    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, writer, epoch)

        current_iou = check_accuracy(val_loader, model, loss_fn, config.DEVICE, writer, epoch)
        scheduler.step(current_iou)

        if config.SAVE_MODEL and current_iou > best_val_iou:
            best_val_iou = current_iou
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename=os.path.join(config.CHECKPOINT_DIR, f"best_model.pth.tar"))

    writer.close()


if __name__ == "__main__":
    main()