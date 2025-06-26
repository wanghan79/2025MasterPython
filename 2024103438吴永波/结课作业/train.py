import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import config
from dataset import get_dataloader
from model import UNet
from loss import CombinedLoss
from metrics import SegmentationMetric
from utils import set_seed, save_checkpoint

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer=None):
    model.train()
    running_loss = 0.0
    metric = SegmentationMetric(num_classes=config.num_classes, ignore_index=config.ignore_index)
    pbar = tqdm(loader, desc=f'Train Epoch {epoch}')
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        metric.update(preds, masks)
        if writer and i % config.print_interval == 0:
            writer.add_scalar('Train/BatchLoss', loss.item(), epoch * len(loader) + i)
        pbar.set_postfix({'loss': loss.item()})
    scores = metric.get_scores()
    mean_loss = running_loss / len(loader)
    if writer:
        writer.add_scalar('Train/EpochLoss', mean_loss, epoch)
        writer.add_scalar('Train/MeanIoU', scores['Mean IoU'], epoch)
    return mean_loss, scores

def validate(model, loader, criterion, device, epoch, writer=None):
    model.eval()
    running_loss = 0.0
    metric = SegmentationMetric(num_classes=config.num_classes, ignore_index=config.ignore_index)
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Val Epoch {epoch}')
        for i, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            metric.update(preds, masks)
    scores = metric.get_scores()
    mean_loss = running_loss / len(loader)
    if writer:
        writer.add_scalar('Val/EpochLoss', mean_loss, epoch)
        writer.add_scalar('Val/MeanIoU', scores['Mean IoU'], epoch)
    return mean_loss, scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.py', help='Config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint')
    args = parser.parse_args()

    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    writer = SummaryWriter(config.log_dir) if config.use_tensorboard else None

    train_loader = get_dataloader(
        config.train_images, config.train_masks,
        input_size=(config.input_height, config.input_width),
        batch_size=config.batch_size, num_workers=config.num_workers, mode='train', augment=True)
    val_loader = get_dataloader(
        config.val_images, config.val_masks,
        input_size=(config.input_height, config.input_width),
        batch_size=config.batch_size, num_workers=config.num_workers, mode='val', augment=False)

    model = UNet(n_channels=3, n_classes=config.num_classes).to(device)
    criterion = CombinedLoss(ignore_index=config.ignore_index)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    start_epoch = 0
    best_miou = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        best_miou = checkpoint.get('best_score', 0)
        print(f'Resumed from {args.resume}, epoch {start_epoch}, best_miou {best_miou}')

    for epoch in range(start_epoch, config.epochs):
        train_loss, train_scores = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        if (epoch + 1) % config.val_interval == 0:
            val_loss, val_scores = validate(model, val_loader, criterion, device, epoch, writer)
            miou = val_scores['Mean IoU']
            is_best = miou > best_miou
            if is_best:
                best_miou = miou
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_score': best_miou
            }, is_best, config.save_dir)
            scheduler.step(miou)
        else:
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_score': best_miou
            }, False, config.save_dir)
    if writer:
        writer.close()

if __name__ == '__main__':
    main() 