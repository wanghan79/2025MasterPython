import os
import cv2
import numpy as np
import random
import time
from collections import defaultdict
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

# ����������ӱ�֤���ظ���
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ���ò���
class Config:
    # ��������
    DATA_DIR = "data"
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    LABEL_DIR = os.path.join(DATA_DIR, "labels")
    CLASS_FILE = os.path.join(DATA_DIR, "classes.names")
    TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
    VAL_TXT = os.path.join(DATA_DIR, "val.txt")
    
    # ģ������
    IMAGE_SIZE = 416
    GRID_SIZES = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
    ANCHORS = [
        [(10, 13), (16, 30), (33, 23)],  # СĿ��
        [(30, 61), (62, 45), (59, 119)],  # ��Ŀ��
        [(116, 90), (156, 198), (373, 326)]  # ��Ŀ��
    ]
    NUM_CLASSES = 20  # VOC���ݼ���20��
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    
    # ѵ������
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    CHECKPOINT_DIR = "checkpoints"
    LOG_INTERVAL = 10
    
    # �豸����
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ȷ������Ŀ¼����
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# ������ǿ��Ԥ����
class ImageTransforms:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image):
        return self.transform(image)

# �Զ������ݼ���
class YOLODataset(Dataset):
    def __init__(self, image_paths, transform=None, train=True):
        self.image_paths = image_paths
        self.transform = transform
        self.train = train
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        # ��ȡ��Ӧ�ı�ǩ�ļ�·��
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
                    
                    # ת��Ϊ�߽������
                    x_min = (x_center - box_width / 2) * width
                    y_min = (y_center - box_height / 2) * height
                    x_max = (x_center + box_width / 2) * width
                    y_max = (y_center + box_height / 2) * height
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(class_id))
        
        # ������ǿ
        if self.train and random.random() > 0.5:
            # ���ˮƽ��ת
            image, boxes = self.random_flip(image, boxes)
        
        # ת��ΪTensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # ת��ΪYOLO��ʽ��Ŀ��
        targets = self.build_targets(boxes, labels)
        
        return image, targets
    
    def random_flip(self, image, boxes):
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            width = image.size[0]
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                new_x_min = width - x_max
                new_x_max = width - x_min
                box[0], box[2] = new_x_min, new_x_max
        return image, boxes
    
    def build_targets(self, boxes, labels):
        """���߽��ͱ�ǩת��ΪYOLO��ʽ��Ŀ��"""
        num_anchors = 3
        targets = [torch.zeros((num_anchors, grid_size, grid_size, 6)) 
                  for grid_size in Config.GRID_SIZES]
        
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            box_width = x_max - x_min
            box_height = y_max - y_min
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            # ����߽�����������ͼ�ı���
            box_ratio = box_width * box_height / (Config.IMAGE_SIZE ** 2)
            
            # ѡ�����ʺϵ�����ͼ�߶�
            if box_ratio > 0.1:  # ��Ŀ��
                grid_idx = 2  # ʹ����������ͼ
            elif box_ratio > 0.01:  # ��Ŀ��
                grid_idx = 1  # ʹ���е�����ͼ
            else:  # СĿ��
                grid_idx = 0  # ʹ����С������ͼ
                
            grid_size = Config.GRID_SIZES[grid_idx]
            stride = Config.IMAGE_SIZE / grid_size
            
            # ������������
            grid_x = int(x_center / stride)
            grid_y = int(y_center / stride)
            
            # ��������������ƫ����
            x_offset = (x_center / stride) - grid_x
            y_offset = (y_center / stride) - grid_y
            
            # �����Ⱥ͸߶������ê��ı���
            box_width /= stride
            box_height /= stride
            
            # ѡ�����ê��
            anchor_indices = Config.ANCHORS[grid_idx]
            best_iou = 0
            best_anchor = 0
            
            for i, (anchor_w, anchor_h) in enumerate(anchor_indices):
                # ����ê��ͱ߽���IOU
                min_w = min(anchor_w, box_width)
                min_h = min(anchor_h, box_height)
                intersection = min_w * min_h
                union = anchor_w * anchor_h + box_width * box_height - intersection
                iou = intersection / union
                
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = i
            
            # ����Ŀ��
            if grid_x < grid_size and grid_y < grid_size:
                targets[grid_idx][best_anchor, grid_y, grid_x, 0] = x_offset
                targets[grid_idx][best_anchor, grid_y, grid_x, 1] = y_offset
                targets[grid_idx][best_anchor, grid_y, grid_x, 2] = box_width
                targets[grid_idx][best_anchor, grid_y, grid_x, 3] = box_height
                targets[grid_idx][best_anchor, grid_y, grid_x, 4] = 1  # ���Ŷ�
                targets[grid_idx][best_anchor, grid_y, grid_x, 5] = label
        
        return targets

# �������ݼ�
def load_dataset():
    with open(Config.TRAIN_TXT, 'r') as f:
        train_images = [line.strip() for line in f.readlines()]
    
    with open(Config.VAL_TXT, 'r') as f:
        val_images = [line.strip() for line in f.readlines()]
    
    transform = ImageTransforms()
    train_dataset = YOLODataset(train_images, transform=transform, train=True)
    val_dataset = YOLODataset(val_images, transform=transform, train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=yolo_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=yolo_collate_fn
    )
    
    return train_loader, val_loader

# �Զ���collate����
def yolo_collate_fn(batch):
    images = []
    targets_list = [[] for _ in range(3)]  # 3���߶ȵ�Ŀ��
    
    for image, targets in batch:
        images.append(image)
        for i in range(3):
            targets_list[i].append(targets[i])
    
    images = torch.stack(images, 0)
    
    # ��Ŀ��ת��Ϊ����
    for i in range(3):
        targets_list[i] = torch.stack(targets_list[i], 0)
    
    return images, targets_list

# Darknet-53�Ļ�����
class DarknetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DarknetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*2)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leaky_relu(out)
        out += residual
        return out

# Darknet-53����������
class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # �²�����
        self.layer1 = self._make_layer(32, 64, num_blocks=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=8)
        self.layer4 = self._make_layer(256, 512, num_blocks=8)
        self.layer5 = self._make_layer(512, 1024, num_blocks=4)
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        # �²������
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.1))
        
        # ��Ӳв��
        for _ in range(num_blocks):
            layers.append(DarknetBlock(out_channels, out_channels//2))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        route1 = self.layer3(x)  # �����е�Ŀ����
        route2 = self.layer4(route1)  # ����СĿ����
        x = self.layer5(route2)  # ���ڴ�Ŀ����
        
        return route1, route2, x

# YOLOv3����
class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.grid_size = 0  # ��ʼ��
        
        # Ԥ��ת������
        self.conv = nn.Conv2d(256, self.num_anchors * (5 + num_classes), kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        batch_size = x.size(0)
        grid_size = x.size(2)
        
        # Ԥ�����
        prediction = self.conv(x)
        prediction = prediction.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()  # (B, A, G, G, 5+C)
        
        # Ӧ��sigmoid��������������Ŷ�
        prediction[..., 0:2] = torch.sigmoid(prediction[..., 0:2])  # x, y
        prediction[..., 4] = torch.sigmoid(prediction[..., 4])  # ���Ŷ�
        prediction[..., 5:] = torch.sigmoid(prediction[..., 5:])  # ������
        
        # �������ѵ��ģʽ�£������������
        if not self.training:
            if grid_size != self.grid_size:
                self.grid_size = grid_size
                self.create_grids(grid_size)
            
            # Ӧ��ê�������ƫ����
            prediction[..., 0] = (prediction[..., 0] + self.grid_x) / grid_size  # x
            prediction[..., 1] = (prediction[..., 1] + self.grid_y) / grid_size  # y
            prediction[..., 2] = torch.exp(prediction[..., 2]) * self.anchor_w  # width
            prediction[..., 3] = torch.exp(prediction[..., 3]) * self.anchor_h  # height
            
            # ���Ż�ԭʼͼ��ߴ�
            prediction[..., :4] *= self.stride
        
        return prediction
    
    def create_grids(self, grid_size):
        self.stride = self.img_dim / grid_size
        
        # ��������ƫ����
        self.grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).float()
        self.grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).float()
        
        # ����ê���Ⱥ͸߶�
        self.anchor_w = torch.Tensor(self.anchors)[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = torch.Tensor(self.anchors)[:, 1:2].view((1, self.num_anchors, 1, 1))
        
        if next(self.parameters()).is_cuda:
            self.grid_x = self.grid_x.cuda()
            self.grid_y = self.grid_y.cuda()
            self.anchor_w = self.anchor_w.cuda()
            self.anchor_h = self.anchor_h.cuda()

# YOLOv3����ģ��
class YOLOv3(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOv3, self).__init__()
        self.backbone = Darknet53()
        self.img_dim = img_dim
        
        # ��Ŀ����·�� (13x13)
        self.detect1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1)
        )
        
        self.yolo1 = YOLOLayer(anchors[2], num_classes, img_dim)
        
        # �е�Ŀ����·�� (26x26)
        self.upsample1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
        self.detect2 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )
        
        self.yolo2 = YOLOLayer(anchors[1], num_classes, img_dim)
        
        # СĿ����·�� (52x52)
        self.upsample2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
        self.detect3 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        
        self.yolo3 = YOLOLayer(anchors[0], num_classes, img_dim)
    
    def forward(self, x):
        # ��������
        route1, route2, x = self.backbone(x)
        
        # ��Ŀ����
        x1 = self.detect1(x)
        yolo1_out = self.yolo1(x1)
        
        # �е�Ŀ����
        x2 = self.upsample1(x1)
        x2 = torch.cat([x2, route2], 1)
        x2 = self.detect2(x2)
        yolo2_out = self.yolo2(x2)
        
        # СĿ����
        x3 = self.upsample2(x2)
        x3 = torch.cat([x3, route1], 1)
        x3 = self.detect3(x3)
        yolo3_out = self.yolo3(x3)
        
        if self.training:
            return [yolo1_out, yolo2_out, yolo3_out]
        else:
            return torch.cat([yolo1_out, yolo2_out, yolo3_out], 1)

# YOLO��ʧ����
class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.num_classes = Config.NUM_CLASSES
        self.anchors = Config.ANCHORS
        self.img_size = Config.IMAGE_SIZE
        self.ignore_threshold = 0.5
    
    def forward(self, predictions, targets):
        # ��ʼ����ʧ
        loss = 0
        mse_loss = 0
        bce_loss = 0
        cls_loss = 0
        
        for i in range(3):  # �����߶�
            # ��ȡԤ���Ŀ��
            pred = predictions[i]
            target = targets[i]
            batch_size = pred.size(0)
            grid_size = pred.size(2)
            stride = self.img_size / grid_size
            
            # ��ȡê��
            anchors = [(a[0]/stride, a[1]/stride) for a in self.anchors[i]]
            
            # ת��Ԥ���ʽ
            pred = pred.view(batch_size, len(anchors), 5 + self.num_classes, grid_size, grid_size)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # ��ȡԤ��ֵ
            pred_x = torch.sigmoid(pred[..., 0])
            pred_y = torch.sigmoid(pred[..., 1])
            pred_w = pred[..., 2]
            pred_h = pred[..., 3]
            pred_conf = torch.sigmoid(pred[..., 4])
            pred_cls = torch.sigmoid(pred[..., 5:])
            
            # ��ȡĿ��ֵ
            target_x = target[..., 0]
            target_y = target[..., 1]
            target_w = target[..., 2]
            target_h = target[..., 3]
            target_conf = target[..., 4]
            target_cls = target[..., 5]
            
            # ��������
            obj_mask = target_conf == 1
            noobj_mask = target_conf == 0
            
            # ����������ʧ
            mse_loss += self.mse_loss(pred_x[obj_mask], target_x[obj_mask])
            mse_loss += self.mse_loss(pred_y[obj_mask], target_y[obj_mask])
            
            # ��������ʧ
            mse_loss += self.mse_loss(pred_w[obj_mask], target_w[obj_mask])
            mse_loss += self.mse_loss(pred_h[obj_mask], target_h[obj_mask])
            
            # �������Ŷ���ʧ
            bce_loss += self.bce_loss(pred_conf[obj_mask], target_conf[obj_mask])
            bce_loss += 0.5 * self.bce_loss(pred_conf[noobj_mask], target_conf[noobj_mask])
            
            # ���������ʧ
            cls_loss += self.bce_loss(pred_cls[obj_mask], F.one_hot(target_cls[obj_mask].long(), self.num_classes).float())
        
        # ����ʧ
        loss = mse_loss + bce_loss + cls_loss
        return loss / batch_size

# ѵ������
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(Config.DEVICE)
        targets = [t.to(Config.DEVICE) for t in targets]
        
        # ǰ�򴫲�
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # ���򴫲�
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if batch_idx % Config.LOG_INTERVAL == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = train_loss / len(train_loader)
    print(f'\nTrain set: Average loss: {avg_loss:.4f}\n')
    return avg_loss

# ��֤����
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.DEVICE)
            targets = [t.to(Config.DEVICE) for t in targets]
            
            outputs = model(images)
            val_loss += criterion(outputs, targets).item()
    
    avg_loss = val_loss / len(val_loader)
    print(f'Validation set: Average loss: {avg_loss:.4f}\n')
    return avg_loss

# �������
def save_checkpoint(model, optimizer, epoch, loss, filename):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, filename)

# ���ؼ���
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

# �Ǽ���ֵ����
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    ��Ԥ�������зǼ���ֵ����
    """
    # ת��Ԥ���ʽ: (����x, ����y, ���, �߶�) -> (����x, ����y, ����x, ����y)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    
    for image_i, image_pred in enumerate(prediction):
        # ���˵������Ŷȵ�Ԥ��
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        
        # ���û��ʣ���Ԥ�⣬������
        if not image_pred.size(0):
            continue
        
        # ����������
        class_conf, class_pred = torch.max(image_pred[:, 5:], 1, keepdim=True)
        
        # ��������� (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        
        # ��ȡͼ���м�⵽�����
        unique_labels = detections[:, -1].cpu().unique()
        
        for c in unique_labels:
            # ��ȡ�ض����ļ����
            detections_class = detections[detections[:, -1] == c]
            
            # �������Ŷ�����
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            
            # ִ�зǼ���ֵ����
            max_detections = []
            while detections_class.size(0):
                # ��ȡ��ǰ������Ŷȵļ����
                max_detections.append(detections_class[0].unsqueeze(0))
                
                # ���ֻʣһ�����������ֹͣ
                if len(detections_class) == 1:
                    break
                
                # ���㵱ǰ������������������IOU
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                
                # �Ƴ�IOU������ֵ�ļ����
                detections_class = detections_class[1:][ious < nms_thres]
            
            max_detections = torch.cat(max_detections).data
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
    
    return output

# ��������: ת���߽���ʽ
def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # ����x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # ����y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # ����x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # ����y
    return y

# ��������: ����IOU
def bbox_iou(box1, box2):
    """
    ��������߽��֮���IOU
    """
    # ��ȡ�߽�������
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # �����ཻ���������
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    # �����ཻ�������
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # ���㲢���������
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    # ����IOU
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

# ��������ӻ�
def plot_detections(image, detections, classes, colors):
    """���Ƽ����"""
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # ���Ʊ߽��ͱ�ǩ
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        # ��ӡ�����
        print(f'+ Label: {classes[int(cls_pred)]}, Conf: {cls_conf.item():.2f}')
        
        # ���ű߽������
        box_w = x2 - x1
        box_h = y2 - y1
        
        # �������ο�
        color = colors[int(cls_pred)]
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(bbox)
        
        # ��ӱ�ǩ
        plt.text(
            x1, y1,
            s=f'{classes[int(cls_pred)]}: {cls_conf.item():.2f}',
            color='white',
            verticalalignment='top',
            bbox={'color': color, 'pad': 0}
        )
    
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.show()

# ������
def main():
    # �������ݼ�
    train_loader, val_loader = load_dataset()
    
    # ��ʼ��ģ��
    model = YOLOv3(Config.ANCHORS, Config.NUM_CLASSES, Config.IMAGE_SIZE).to(Config.DEVICE)
    
    # �����Ż�������ʧ����
    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = YOLOLoss()
    
    # ѵ��ѭ��
    best_loss = float('inf')
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, epoch)
        val_loss = validate(model, val_loader, criterion)
        
        # �������ģ��
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(Config.CHECKPOINT_DIR, f'yolov3_epoch{epoch}_loss{val_loss:.2f}.pth')
            )
    
    print("ѵ�����!")

# ������
def detect():
    # ����������ɫ
    classes = load_classes(Config.CLASS_FILE)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(classes))]
    
    # ��ʼ��ģ��
    model = YOLOv3(Config.ANCHORS, Config.NUM_CLASSES, Config.IMAGE_SIZE).to(Config.DEVICE)
    
    # ����Ԥѵ��Ȩ��
    checkpoint = torch.load(os.path.join(Config.CHECKPOINT_DIR, 'yolov3_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # ���ز���ͼ��
    image_path = "test.jpg"
    image = Image.open(image_path).convert('RGB')
    image_tensor = transforms.ToTensor()(image)
    image_tensor = torch.unsqueeze(image_tensor, 0).to(Config.DEVICE)
    
    # ִ�м��
    with torch.no_grad():
        predictions = model(image_tensor)
        detections = non_max_suppression(predictions, Config.SCORE_THRESHOLD, Config.IOU_THRESHOLD)
    
    # ���ӻ����
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plot_detections(image, detections[0], classes, colors)

# ��������: �������
def load_classes(path):
    with open(path, 'r') as f:
        names = [line.strip() for line in f.readlines()]
    return names

if __name__ == '__main__':
    main()
    # detect()  # ѵ����ɺ�ȡ��ע�������м��
