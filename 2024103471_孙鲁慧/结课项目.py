import os
import cv2
import numpy as np
import torch
import yaml
import random
import time
import shutil
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ExifTags
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# YOLOv8模型相关导入
try:
    from ultralytics import YOLO
    from ultralytics.yolo.data.augment import LetterBox
    from ultralytics.yolo.engine.results import Results
    from ultralytics.yolo.utils import ops, DEFAULT_CFG
    from ultralytics.yolo.utils.checks import check_requirements, check_yaml
    from ultralytics.yolo.utils.plotting import Annotator, colors
except ImportError:
    print("请先安装ultralytics库: pip install ultralytics")
    raise


# 小目标检测增强模块
class SmallObjectDetectionEnhancer:
    """小目标检测增强类，包含多种专门针对小目标检测的增强方法"""

    def __init__(self,
                 mosaic_prob=0.3,
                 mixup_prob=0.2,
                 cutout_prob=0.1,
                 high_resolution=True,
                 multi_scale_training=True,
                 anchor_free=False):
        """
        初始化小目标检测增强器

        参数:
            mosaic_prob: 使用马赛克数据增强的概率
            mixup_prob: 使用混合增强的概率
            cutout_prob: 使用Cutout增强的概率
            high_resolution: 是否使用高分辨率训练
            multi_scale_training: 是否使用多尺度训练
            anchor_free: 是否使用无锚点检测方法
        """
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.cutout_prob = cutout_prob
        self.high_resolution = high_resolution
        self.multi_scale_training = multi_scale_training
        self.anchor_free = anchor_free

        # 高分辨率设置
        self.high_res_size = 1280  # 高分辨率训练时的尺寸
        self.base_res_size = 640  # 基础分辨率尺寸

        # 多尺度训练设置
        self.scales = [512, 640, 768, 896, 1024]  # 训练时使用的不同尺度
        self.scale_step = 32  # 尺度步长，必须是模型步长的倍数

        # 小目标定义 (面积占比)
        self.small_object_threshold = 0.01  # 小于图像面积1%的目标视为小目标

    def apply_data_augmentation(self, imgs, labels):
        """
        应用针对小目标优化的数据增强

        参数:
            imgs: 图像列表
            labels: 标签列表

        返回:
            增强后的图像和标签
        """
        # 随机决定是否应用马赛克增强
        if random.random() < self.mosaic_prob:
            imgs, labels = self._mosaic_augmentation(imgs, labels)

        # 随机决定是否应用MixUp增强
        if random.random() < self.mixup_prob and len(imgs) > 1:
            imgs, labels = self._mixup_augmentation(imgs, labels)

        # 随机决定是否应用Cutout增强
        if random.random() < self.cutout_prob:
            imgs, labels = self._cutout_augmentation(imgs, labels)

        return imgs, labels

    def _mosaic_augmentation(self, imgs, labels):
        """
        马赛克数据增强 - 将4张图像拼接成一张，有助于小目标检测

        参数:
            imgs: 图像列表
            labels: 标签列表

        返回:
            增强后的图像和标签
        """
        # 确保有足够的图像进行拼接
        if len(imgs) < 4:
            return imgs, labels

        # 随机选择4张图像
        indices = random.sample(range(len(imgs)), 4)
        img4, labels4 = [], []

        # 确定拼接中心点
        h, w = self.high_res_size if self.high_resolution else self.base_res_size, \
               self.high_res_size if self.high_resolution else self.base_res_size
        xc, yc = int(random.uniform(0.3 * w, 0.7 * w)), int(random.uniform(0.3 * h, 0.7 * h))

        # 创建拼接后的图像和标签
        img_mosaic = np.zeros((h, w, 3), dtype=np.uint8)
        labels_mosaic = []

        for i, index in enumerate(indices):
            img, label = imgs[index], labels[index]
            if img is None or len(label) == 0:
                continue

            hi, wi = img.shape[:2]

            # 计算图像在马赛克中的位置
            if i == 0:  # 左上
                x1a, y1a, x2a, y2a = max(xc - wi, 0), max(yc - hi, 0), xc, yc
                x1b, y1b, x2b, y2b = wi - (x2a - x1a), hi - (y2a - y1a), wi, hi
            elif i == 1:  # 右上
                x1a, y1a, x2a, y2a = xc, max(yc - hi, 0), min(xc + wi, w), yc
                x1b, y1b, x2b, y2b = 0, hi - (y2a - y1a), min(wi, x2a - x1a), hi
            elif i == 2:  # 左下
                x1a, y1a, x2a, y2a = max(xc - wi, 0), yc, xc, min(yc + hi, h)
                x1b, y1b, x2b, y2b = wi - (x2a - x1a), 0, wi, min(hi, y2a - y1a)
            elif i == 3:  # 右下
                x1a, y1a, x2a, y2a = xc, yc, min(xc + wi, w), min(yc + hi, h)
                x1b, y1b, x2b, y2b = 0, 0, min(wi, x2a - x1a), min(hi, y2a - y1a)

            # 将图像的一部分复制到马赛克图像中
            img_mosaic[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # 调整标签坐标
            for l in label:
                cls, x, y, w_box, h_box = l
                x_center = x * wi + padw
                y_center = y * hi + padh
                w_box = w_box * wi
                h_box = h_box * hi

                # 计算归一化后的坐标
                x_center /= w
                y_center /= h
                w_box /= w
                h_box /= h

                # 过滤掉超出边界的目标
                if 0 < x_center < 1 and 0 < y_center < 1 and 0 < w_box < 1 and 0 < h_box < 1:
                    labels_mosaic.append([cls, x_center, y_center, w_box, h_box])

        return [img_mosaic], [np.array(labels_mosaic)]

    def _mixup_augmentation(self, imgs, labels):
        """
        MixUp数据增强 - 将两张图像按一定比例混合，有助于小目标检测

        参数:
            imgs: 图像列表
            labels: 标签列表

        返回:
            增强后的图像和标签
        """
        # 随机选择两张图像
        idx1, idx2 = random.sample(range(len(imgs)), 2)
        img1, label1 = imgs[idx1], labels[idx1]
        img2, label2 = imgs[idx2], labels[idx2]

        if img1 is None or img2 is None or len(label1) == 0 or len(label2) == 0:
            return [img1], [label1]

        # 调整图像大小
        h, w = self.high_res_size if self.high_resolution else self.base_res_size, \
               self.high_res_size if self.high_resolution else self.base_res_size

        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))

        # 随机混合比例
        alpha = random.uniform(0.4, 0.6)

        # 混合图像
        img_mix = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

        # 合并标签
        labels_mix = []
        for l in label1:
            labels_mix.append([l[0], l[1], l[2], l[3], l[4]])

        for l in label2:
            labels_mix.append([l[0], l[1], l[2], l[3], l[4]])

        return [img_mix], [np.array(labels_mix)]

    def _cutout_augmentation(self, imgs, labels):
        """
        Cutout数据增强 - 随机遮挡图像的一部分，有助于模型关注小目标

        参数:
            imgs: 图像列表
            labels: 标签列表

        返回:
            增强后的图像和标签
        """
        # 只处理第一张图像
        img, label = imgs[0], labels[0]
        if img is None or len(label) == 0:
            return imgs, labels

        h, w = img.shape[:2]

        # 随机决定遮挡块的数量和大小
        num_blocks = random.randint(3, 8)
        block_size = random.randint(int(min(h, w) * 0.05), int(min(h, w) * 0.15))

        # 随机遮挡
        for _ in range(num_blocks):
            x = random.randint(0, w - block_size)
            y = random.randint(0, h - block_size)
            img[y:y + block_size, x:x + block_size] = 0  # 用黑色遮挡

        return [img], [label]

    def get_training_resolution(self):
        """获取训练时使用的分辨率，支持多尺度训练"""
        if not self.multi_scale_training:
            return self.high_res_size if self.high_resolution else self.base_res_size

        # 随机选择一个尺度
        return random.choice(self.scales)

    def analyze_dataset_for_small_objects(self, dataset):
        """
        分析数据集中小目标的比例

        参数:
            dataset: 数据集

        返回:
            小目标比例和统计信息
        """
        small_object_count = 0
        total_object_count = 0
        small_object_classes = defaultdict(int)
        class_count = defaultdict(int)

        for img_path, label_path in dataset:
            # 读取标签
            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls = int(parts[0])
                x, y, w_box, h_box = map(float, parts[1:5])

                # 计算目标面积占比
                area_ratio = w_box * h_box

                total_object_count += 1
                class_count[cls] += 1

                if area_ratio < self.small_object_threshold:
                    small_object_count += 1
                    small_object_classes[cls] += 1

        # 计算小目标比例
        small_object_ratio = small_object_count / total_object_count if total_object_count > 0 else 0

        # 计算每个类别的小目标比例
        small_object_class_ratios = {}
        for cls in class_count:
            if class_count[cls] > 0:
                small_object_class_ratios[cls] = small_object_classes[cls] / class_count[cls]

        return {
            'small_object_ratio': small_object_ratio,
            'small_object_count': small_object_count,
            'total_object_count': total_object_count,
            'small_object_class_ratios': small_object_class_ratios
        }


# 小目标检测评估器
class SmallObjectEvaluator:
    """小目标检测评估器，提供专门针对小目标的评估指标"""

    def __init__(self, small_object_threshold=0.01):
        """
        初始化小目标评估器

        参数:
            small_object_threshold: 定义小目标的阈值(相对于图像面积的比例)
        """
        self.small_object_threshold = small_object_threshold

    def evaluate(self, pred_boxes, gt_boxes, image_size, iou_threshold=0.5):
        """
        评估检测结果，分别计算所有目标和小目标的指标

        参数:
            pred_boxes: 预测边界框 [x_center, y_center, width, height, confidence, class]
            gt_boxes: 真实边界框 [x_center, y_center, width, height, class]
            image_size: 图像尺寸 (width, height)
            iou_threshold: IoU阈值

        返回:
            评估结果字典
        """
        img_width, img_height = image_size
        img_area = img_width * img_height

        # 将预测框和真实框分为小目标和所有目标
        pred_boxes_all = pred_boxes
        gt_boxes_all = gt_boxes

        pred_boxes_small = []
        gt_boxes_small = []

        for box in pred_boxes:
            x_center, y_center, w_box, h_box = box[:4]
            box_area = w_box * h_box * img_area
            if box_area / img_area < self.small_object_threshold:
                pred_boxes_small.append(box)

        for box in gt_boxes:
            x_center, y_center, w_box, h_box = box[:4]
            box_area = w_box * h_box * img_area
            if box_area / img_area < self.small_object_threshold:
                gt_boxes_small.append(box)

        # 计算所有目标的指标
        metrics_all = self._calculate_metrics(pred_boxes_all, gt_boxes_all, iou_threshold)

        # 计算小目标的指标
        metrics_small = self._calculate_metrics(pred_boxes_small, gt_boxes_small, iou_threshold)

        return {
            'all_objects': metrics_all,
            'small_objects': metrics_small,
            'small_object_ratio': len(gt_boxes_small) / len(gt_boxes_all) if len(gt_boxes_all) > 0 else 0
        }

    def _calculate_metrics(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """
        计算评估指标

        参数:
            pred_boxes: 预测边界框
            gt_boxes: 真实边界框
            iou_threshold: IoU阈值

        返回:
            包含各种指标的字典
        """
        if not pred_boxes or not gt_boxes:
            return {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'tp': 0,
                'fp': 0,
                'fn': 0
            }

        # 按置信度排序预测框
        pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

        # 初始化匹配结果
        gt_matched = [False] * len(gt_boxes)
        tp = 0
        fp = 0

        # 计算TP和FP
        for pred in pred_boxes:
            best_iou = 0
            best_idx = -1

            for i, gt in enumerate(gt_boxes):
                # 检查类别是否匹配
                if int(pred[5]) != int(gt[4]):
                    continue

                # 计算IoU
                iou = self._calculate_iou(pred[:4], gt[:4])

                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            # 判断是否为TP
            if best_iou >= iou_threshold and not gt_matched[best_idx]:
                tp += 1
                gt_matched[best_idx] = True
            else:
                fp += 1

        # 计算FN
        fn = len(gt_boxes) - sum(gt_matched)

        # 计算精确率、召回率和F1分数
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的IoU

        参数:
            box1: 第一个边界框 [x_center, y_center, width, height]
            box2: 第二个边界框 [x_center, y_center, width, height]

        返回:
            IoU值
        """
        # 转换为坐标形式 [x1, y1, x2, y2]
        x1_1, y1_1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        x2_1, y2_1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2

        x1_2, y1_2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        x2_2, y2_2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

        # 计算交集坐标
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        # 计算交集面积
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        # 计算并集面积
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        # 计算IoU
        iou = inter_area / union_area if union_area > 0 else 0

        return iou

    def visualize_detections(self, image, pred_boxes, gt_boxes, image_size, class_names=None):
        """
        可视化检测结果，区分小目标和大目标

        参数:
            image: 图像
            pred_boxes: 预测边界框
            gt_boxes: 真实边界框
            image_size: 图像尺寸
            class_names: 类别名称列表

        返回:
            可视化后的图像
        """
        img_width, img_height = image_size
        img_area = img_width * img_height

        # 复制图像
        vis_image = image.copy()

        # 可视化预测框
        for box in pred_boxes:
            x_center, y_center, w_box, h_box, conf, cls = box
            box_area = w_box * h_box * img_area

            # 转换为像素坐标
            x1 = int((x_center - w_box / 2) * img_width)
            y1 = int((y_center - h_box / 2) * img_height)
            x2 = int((x_center + w_box / 2) * img_width)
            y2 = int((y_center + h_box / 2) * img_height)

            # 区分小目标和大目标的颜色
            color = (0, 255, 0) if box_area / img_area < self.small_object_threshold else (0, 0, 255)

            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # 添加标签
            label = f"{class_names[int(cls)] if class_names else str(int(cls))} {conf:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 可视化真实框
        for box in gt_boxes:
            x_center, y_center, w_box, h_box, cls = box
            box_area = w_box * h_box * img_area

            # 转换为像素坐标
            x1 = int((x_center - w_box / 2) * img_width)
            y1 = int((y_center - h_box / 2) * img_height)
            x2 = int((x_center + w_box / 2) * img_width)
            y2 = int((y_center + h_box / 2) * img_height)

            # 区分小目标和大目标的颜色
            color = (0, 165, 255) if box_area / img_area < self.small_object_threshold else (255, 0, 0)

            # 绘制边界框(虚线)
            for i in range(0, abs(x2 - x1), 10):
                if x1 + i < x2:
                    cv2.line(vis_image, (x1 + i, y1), (x1 + i + 5, y1), color, 2)
                    cv2.line(vis_image, (x1 + i, y2), (x1 + i + 5, y2), color, 2)
            for i in range(0, abs(y2 - y1), 10):
                if y1 + i < y2:
                    cv2.line(vis_image, (x1, y1 + i), (x1, y1 + i + 5), color, 2)
                    cv2.line(vis_image, (x2, y1 + i), (x2, y1 + i + 5), color, 2)

            # 添加标签
            label = f"{class_names[int(cls)] if class_names else str(int(cls))}"
            cv2.putText(vis_image, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis_image


# 小目标检测训练器
class SmallObjectDetector:
    """基于YOLOv8的小目标检测训练器"""

    def __init__(self, model_name='yolov8n.pt', data_config=None, device=None):
        """
        初始化小目标检测器

        参数:
            model_name: 预训练模型名称
            data_config: 数据配置文件路径
            device: 训练设备，如'cuda'或'cpu'
        """
        self.model_name = model_name
        self.data_config = data_config
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

        # 初始化模型
        self.model = YOLO(model_name)

        # 初始化增强器
        self.enhancer = SmallObjectDetectionEnhancer(
            mosaic_prob=0.5,
            mixup_prob=0.2,
            cutout_prob=0.1,
            high_resolution=True,
            multi_scale_training=True,
            anchor_free=False
        )

        # 初始化评估器
        self.evaluator = SmallObjectEvaluator(small_object_threshold=0.01)

        # 存储训练历史
        self.train_history = {
            'all_objects': {
                'precision': [],
                'recall': [],
                'f1': []
            },
            'small_objects': {
                'precision': [],
                'recall': [],
                'f1': []
            },
            'loss': []
        }

        # 确保模型在指定设备上
        self.model.to(self.device)

    def prepare_dataset(self, data_dir, val_split=0.2, test_split=0.1):
        """
        准备数据集，包括划分训练集、验证集和测试集

        参数:
            data_dir: 数据集目录
            val_split: 验证集比例
            test_split: 测试集比例

        返回:
            划分后的数据集
        """
        # 读取图像和标签文件
        image_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')

        # 获取所有图像文件
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            image_files.extend(Path(image_dir).glob(f'**/*.{ext}'))

        # 构建图像-标签对
        dataset = []
        for img_path in image_files:
            img_name = img_path.stem
            label_path = os.path.join(label_dir, f'{img_name}.txt')

            if os.path.exists(label_path):
                dataset.append((str(img_path), str(label_path)))

        # 划分数据集
        train_val, test = train_test_split(dataset, test_size=test_split, random_state=42)
        train, val = train_test_split(train_val, test_size=val_split / (1 - test_split), random_state=42)

        return train, val, test

    def analyze_dataset(self, dataset):
        """
        分析数据集中小目标的分布情况

        参数:
            dataset: 数据集
        """
        print("正在分析数据集中的小目标分布...")
        stats = self.enhancer.analyze_dataset_for_small_objects(dataset)

        print(f"小目标占比: {stats['small_object_ratio']:.2%}")
        print(f"小目标数量: {stats['small_object_count']}")
        print(f"总目标数量: {stats['total_object_count']}")

        print("\n各类别小目标比例:")
        for cls, ratio in stats['small_object_class_ratios'].items():
            print(f"类别 {cls}: {ratio:.2%}")

        return stats

    def train(self,
              epochs=100,
              batch_size=16,
              imgsz=640,
              lr0=0.01,
              lrf=0.01,
              patience=50,
              pretrained=True):
        """
        训练模型

        参数:
            epochs: 训练轮数
            batch_size: 批次大小
            imgsz: 输入图像尺寸
            lr0: 初始学习率
            lrf: 最终学习率
            patience: 早停耐心值
            pretrained: 是否使用预训练模型
        """
        print(f"开始训练小目标检测模型，设备: {self.device}")

        # 设置训练参数
        train_args = {
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'lr0': lr0,
            'lrf': lrf,
            'patience': patience,
            'pretrained': pretrained,
            'device': self.device,
            'data': self.data_config,
            'project': 'runs/detect',
            'name': f'small_object_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': False,
            'verbose': True,
            'cache': True,
            'optimizer': 'Adam',
            'cos_lr': True,
            'rect': False,
            'close_mosaic': 10  # 最后10个epoch关闭马赛克增强
        }

        # 训练模型
        self.model.train(**train_args)

        # 保存训练历史
        self._save_training_history()

        print("训练完成!")

    def _save_training_history(self):
        """保存训练历史记录"""
        # 从模型中提取训练历史
        if hasattr(self.model, 'history') and self.model.history:
            # 提取损失和指标
            for k, v in self.model.history.items():
                if k.startswith('metrics/'):
                    metric_name = k.split('/')[1]
                    if metric_name in ['precision', 'recall', 'mAP50', 'mAP50-95']:
                        self.train_history['all_objects'][metric_name] = v

            # 提取损失
            if 'train/box_loss' in self.model.history:
                self.train_history['loss'] = [sum(x) for x in zip(
                    self.model.history['train/box_loss'],
                    self.model.history.get('train/cls_loss', [0] * len(self.model.history['train/box_loss'])),
                    self.model.history.get('train/dfl_loss', [0] * len(self.model.history['train/box_loss']))
                )]

    def evaluate(self, dataset, batch_size=8, imgsz=640, conf=0.25, iou=0.5):
        """
        评估模型性能

        参数:
            dataset: 评估数据集
            batch_size: 批次大小
            imgsz: 输入图像尺寸
            conf: 置信度阈值
            iou: IoU阈值

        返回:
            评估结果
        """
        print("开始评估模型性能...")

        # 创建评估器
        evaluator = SmallObjectEvaluator(small_object_threshold=0.01)

        # 存储所有结果
        all_results = []

        # 分批处理数据
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i + batch_size]

            # 准备输入数据
            images = []
            gt_boxes_batch = []
            image_sizes = []

            for img_path, label_path in batch:
                # 读取图像
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                image_sizes.append((w, h))

                # 调整图像大小
                img = cv2.resize(img, (imgsz, imgsz))
                images.append(img)

                # 读取标签
                boxes = []
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        cls = int(parts[0])
                        x_center, y_center, box_w, box_h = map(float, parts[1:5])

                        # 归一化到原始图像尺寸
                        x_center_orig = x_center
                        y_center_orig = y_center
                        box_w_orig = box_w
                        box_h_orig = box_h

                        boxes.append([x_center_orig, y_center_orig, box_w_orig, box_h_orig, cls])

                gt_boxes_batch.append(boxes)

            # 模型推理
            results = self.model(images, conf=conf, iou=iou)

            # 处理推理结果
            for j, result in enumerate(results):
                pred_boxes = []
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf_score = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        # 转换为中心点坐标和宽高
                        x_center = (x1 + x2) / 2 / imgsz
                        y_center = (y1 + y2) / 2 / imgsz
                        box_w = (x2 - x1) / imgsz
                        box_h = (y2 - y1) / imgsz

                        pred_boxes.append([x_center, y_center, box_w, box_h, conf_score, cls])

                # 评估当前图像
                eval_result = evaluator.evaluate(
                    pred_boxes,
                    gt_boxes_batch[j],
                    image_sizes[j],
                    iou_threshold=iou
                )

                all_results.append(eval_result)

        # 计算平均结果
        avg_results = {
            'all_objects': {
                'precision': np.mean([r['all_objects']['precision'] for r in all_results]),
                'recall': np.mean([r['all_objects']['recall'] for r in all_results]),
                'f1': np.mean([r['all_objects']['f1'] for r in all_results])
            },
            'small_objects': {
                'precision': np.mean([r['small_objects']['precision'] for r in all_results]),
                'recall': np.mean([r['small_objects']['recall'] for r in all_results]),
                'f1': np.mean([r['small_objects']['f1'] for r in all_results])
            },
            'small_object_ratio': np.mean([r['small_object_ratio'] for r in all_results])
        }

        print("\n评估结果:")
        print("所有目标:")
        print(f"  精确率: {avg_results['all_objects']['precision']:.4f}")
        print(f"  召回率: {avg_results['all_objects']['recall']:.4f}")
        print(f"  F1分数: {avg_results['all_objects']['f1']:.4f}")

        print("\n小目标:")
        print(f"  精确率: {avg_results['small_objects']['precision']:.4f}")
        print(f"  召回率: {avg_results['small_objects']['recall']:.4f}")
        print(f"  F1分数: {avg_results['small_objects']['f1']:.4f}")

        print(f"\n数据集中小目标比例: {avg_results['small_object_ratio']:.2%}")

        return avg_results

    def detect(self, image_path, conf=0.25, iou=0.5, save_path=None):
        """
        对单张图像进行小目标检测

        参数:
            image_path: 图像路径
            conf: 置信度阈值
            iou: IoU阈值
            save_path: 保存路径，若为None则不保存

        返回:
            检测结果和可视化图像
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None, None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 模型推理
        results = self.model(image, conf=conf, iou=iou)

        # 处理检测结果
        pred_boxes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_score = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                # 转换为中心点坐标和宽高(归一化)
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                box_w = (x2 - x1) / w
                box_h = (y2 - y1) / h

                pred_boxes.append([x_center, y_center, box_w, box_h, conf_score, cls])

        # 获取类别名称
        class_names = self.model.names if hasattr(self.model, 'names') else None

        # 可视化检测结果
        vis_image = self.visualize_results(image, pred_boxes, class_names)

        # 保存结果
        if save_path:
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, vis_image_bgr)
            print(f"检测结果已保存至: {save_path}")

        return results, vis_image

    def visualize_results(self, image, pred_boxes, class_names=None):
        """
        可视化检测结果，区分小目标和大目标

        参数:
            image: 原始图像
            pred_boxes: 预测边界框
            class_names: 类别名称列表

        返回:
            可视化后的图像
        """
        h, w = image.shape[:2]
        img_area = h * w

        # 复制图像
        vis_image = image.copy()

        # 小目标阈值
        small_object_threshold = 0.01 * img_area

        # 可视化预测框
        for box in pred_boxes:
            x_center, y_center, w_box, h_box, conf, cls = box

            # 转换为像素坐标
            x_center_px = int(x_center * w)
            y_center_px = int(y_center * h)
            w_box_px = int(w_box * w)
            h_box_px = int(h_box * h)

            x1 = max(0, x_center_px - w_box_px // 2)
            y1 = max(0, y_center_px - h_box_px // 2)
            x2 = min(w, x_center_px + w_box_px // 2)
            y2 = min(h, y_center_px + h_box_px // 2)

            # 计算目标面积
            box_area = (x2 - x1) * (y2 - y1)

            # 区分小目标和大目标的颜色
            color = (0, 255, 0) if box_area < small_object_threshold else (0, 0, 255)

            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # 添加标签
            label = f"{class_names[int(cls)] if class_names else str(int(cls))} {conf:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis_image

    def export(self, format='onnx', imgsz=640):
        """
        导出模型到不同格式

        参数:
            format: 导出格式，如'onnx', 'tensorrt', 'tflite'等
            imgsz: 输入图像尺寸

        返回:
            导出结果
        """
        print(f"正在导出模型到 {format} 格式...")

        # 导出模型
        results = self.model.export(format=format, imgsz=imgsz)

        print(f"模型已导出至: {results}")
        return results

    def plot_training_history(self, save_path=None):
        """
        绘制训练历史曲线

        参数:
            save_path: 保存路径，若为None则不保存
        """
        if not self.train_history['all_objects']['precision']:
            print("没有训练历史记录可绘制")
            return

        epochs = len(self.train_history['all_objects']['precision'])

        # 创建图表
        plt.figure(figsize=(15, 10))

        # 绘制精确率、召回率和F1分数
        plt.subplot(2, 1, 1)
        plt.plot(range(epochs), self.train_history['all_objects']['precision'], 'b-', label='所有目标精确率')
        plt.plot(range(epochs), self.train_history['all_objects']['recall'], 'g-', label='所有目标召回率')
        plt.plot(range(epochs), self.train_history['all_objects']['f1'], 'r-', label='所有目标F1分数')

        if self.train_history['small_objects']['precision']:
            plt.plot(range(epochs), self.train_history['small_objects']['precision'], 'b--', label='小目标精确率')
            plt.plot(range(epochs), self.train_history['small_objects']['recall'], 'g--', label='小目标召回率')
            plt.plot(range(epochs), self.train_history['small_objects']['f1'], 'r--', label='小目标F1分数')

        plt.xlabel('轮次')
        plt.ylabel('指标值')
        plt.title('模型性能指标')
        plt.legend()
        plt.grid(True)

        # 绘制损失曲线
        plt.subplot(2, 1, 2)
        plt.plot(range(epochs), self.train_history['loss'], 'c-', label='总损失')
        plt.xlabel('轮次')
        plt.ylabel('损失值')
        plt.title('训练损失')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # 保存图表
        if save_path:
            plt.savefig(save_path)
            print(f"训练历史图表已保存至: {save_path}")

        plt.show()


# 主函数
def main():
    """主函数，用于命令行调用"""
    parser = argparse.ArgumentParser(description='基于YOLOv8的小目标检测')
    parser.add_argument('--data', type=str, required=True, help='数据配置文件路径')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='预训练模型名称')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default=None, help='训练设备，如"cuda"或"cpu"')
    parser.add_argument('--conf', type=float, default=0.25, help='检测置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5, help='NMS的IoU阈值')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'detect', 'evaluate', 'export'],
                        help='运行模式')
    parser.add_argument('--source', type=str, default=None, help='检测源，图像或视频路径')
    parser.add_argument('--weights', type=str, default=None, help='模型权重路径')
    parser.add_argument('--export_format', type=str, default='onnx', help='导出格式')

    args = parser.parse_args()

    # 检查依赖
    check_requirements('ultralytics')

    # 创建小目标检测器
    detector = SmallObjectDetector(
        model_name=args.model if args.weights is None else args.weights,
        data_config=args.data,
        device=args.device
    )

    if args.mode == 'train':
        # 准备数据集
        data_dir = os.path.dirname(os.path.abspath(args.data))
        train_data, val_data, test_data = detector.prepare_dataset(data_dir)

        # 分析数据集
        detector.analyze_dataset(train_data)

        # 训练模型
        detector.train(
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz
        )

        # 评估模型
        detector.evaluate(test_data, batch_size=args.batch, imgsz=args.imgsz, conf=args.conf, iou=args.iou)

        # 绘制训练历史
        detector.plot_training_history('training_history.png')

    elif args.mode == 'detect':
        if args.source is None:
            print("检测模式需要指定--source参数")
            return

        # 检测单张图像
        results, vis_image = detector.detect(
            image_path=args.source,
            conf=args.conf,
            iou=args.iou,
            save_path='detection_result.jpg'
        )

        # 显示检测结果
        if vis_image is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(vis_image)
            plt.axis('off')
            plt.show()

    elif args.mode == 'evaluate':
        # 准备数据集
        data_dir = os.path.dirname(os.path.abspath(args.data))
        _, _, test_data = detector.prepare_dataset(data_dir)

        # 评估模型
        detector.evaluate(test_data, batch_size=args.batch, imgsz=args.imgsz, conf=args.conf, iou=args.iou)

    elif args.mode == 'export':
        # 导出模型
        detector.export(format=args.export_format, imgsz=args.imgsz)


if __name__ == "__main__":
    main()