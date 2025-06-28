import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import glob
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from lxml import etree
from PIL import Image
import argparse
import logging
import time
import json
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, classification_report

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("object_localization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="图像定位与分类系统")
    parser.add_argument('--data-dir', type=str, default='dataset', 
                        help='数据集根目录')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=15,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--image-size', type=int, default=224,
                        help='输入图像尺寸')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='类别数量')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--eval-only', action='store_true',
                        help='仅运行评估模式')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化预测结果')
    return parser.parse_args()

# 配置类
class Config:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.image_dir = os.path.join(self.data_dir, "images")
        self.annotation_dir = os.path.join(self.data_dir, "annotations", "xmls")
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.image_size = args.image_size
        self.num_classes = args.num_classes
        self.resume = args.resume
        self.eval_only = args.eval_only
        self.visualize = args.visualize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = {0: 'cat', 1: 'dog'}
        self.name_to_id = {'cat': 0, 'dog': 1}
        
        # 创建输出目录
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"配置参数: {vars(args)}")

# 数据预处理工具
class DataPreprocessor:
    """处理图像和标注数据的工具类"""
    
    def __init__(self, config):
        self.config = config
        self.scale = config.image_size
        
    def parse_xml_annotation(self, xml_path):
        """解析XML标注文件"""
        try:
            xml = open(xml_path, 'r', encoding='utf-8').read()
            sel = etree.HTML(xml)
            
            name = sel.xpath('//object/name/text()')[0].lower()
            width = int(sel.xpath('//size/width/text()')[0])
            height = int(sel.xpath('//size/height/text()')[0])
            xmin = int(sel.xpath('//bndbox/xmin/text()')[0])
            ymin = int(sel.xpath('//bndbox/ymin/text()')[0])
            xmax = int(sel.xpath('//bndbox/xmax/text()')[0])
            ymax = int(sel.xpath('//bndbox/ymax/text()')[0])
            
            # 转换为相对坐标
            xmin_rel = xmin / width
            ymin_rel = ymin / height
            xmax_rel = xmax / width
            ymax_rel = ymax / height
            
            class_id = self.config.name_to_id.get(name, -1)
            if class_id == -1:
                logger.warning(f"未知类别: {name} in {xml_path}")
            
            return (xmin_rel, ymin_rel, xmax_rel, ymax_rel, class_id)
        
        except Exception as e:
            logger.error(f"解析XML失败: {xml_path}, 错误: {e}")
            return None
    
    def get_image_paths(self):
        """获取所有图像路径和对应的标注路径"""
        # 获取所有标注文件
        xml_paths = glob.glob(os.path.join(self.config.annotation_dir, '*.xml'))
        logger.info(f"找到 {len(xml_paths)} 个标注文件")
        
        # 提取标注文件名（不含扩展名）
        xml_names = [os.path.splitext(os.path.basename(x))[0] for x in xml_paths]
        
        # 构建对应的图像路径
        image_paths = [os.path.join(self.config.image_dir, f"{name}.jpg") for name in xml_names]
        
        # 过滤不存在的图像
        valid_pairs = [(img, xml) for img, xml in zip(image_paths, xml_paths) if os.path.exists(img)]
        
        logger.info(f"有效图像-标注对: {len(valid_pairs)}")
        return valid_pairs
    
    def create_datasets(self):
        """创建训练集和测试集"""
        # 获取所有有效的图像-标注对
        pairs = self.get_image_paths()
        
        # 打乱顺序
        np.random.seed(42)
        indices = np.random.permutation(len(pairs))
        
        # 划分训练集和测试集 (80% 训练, 20% 测试)
        split_idx = int(len(pairs) * 0.8)
        
        train_pairs = [pairs[i] for i in indices[:split_idx]]
        test_pairs = [pairs[i] for i in indices[split_idx:]]
        
        # 解析标注
        train_labels = [self.parse_xml_annotation(xml) for img, xml in train_pairs]
        test_labels = [self.parse_xml_annotation(xml) for img, xml in test_pairs]
        
        # 过滤无效标注
        train_data = [(img, label) for (img, xml), label in zip(train_pairs, train_labels) if label is not None]
        test_data = [(img, label) for (img, xml), label in zip(test_pairs, test_labels) if label is not None]
        
        logger.info(f"训练集大小: {len(train_data)}")
        logger.info(f"测试集大小: {len(test_data)}")
        
        return train_data, test_data

# 自定义数据集类
class OxfordDataset(data.Dataset):
    """牛津宠物数据集加载器"""
    
    def __init__(self, data, transform=None, augment=False):
        """
        参数:
            data: 包含(图像路径, 标签)的列表
            transform: 图像转换函数
            augment: 是否应用数据增强
        """
        self.data = data
        self.transform = transform
        self.augment = augment
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(self.transform.transforms[0].size[0], scale=(0.8, 1.0)),
        ]) if augment else None
        
        # 统计类别分布
        self.class_distribution = self._compute_class_distribution()
        
    def _compute_class_distribution(self):
        """计算类别分布"""
        class_counts = {0: 0, 1: 0}
        for _, label in self.data:
            class_id = int(label[4])
            if class_id in class_counts:
                class_counts[class_id] += 1
        return class_counts
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, label = self.data[index]
        
        try:
            # 加载图像
            img = Image.open(img_path)
            img = img.convert('RGB')
            
            # 应用数据增强
            if self.augment and self.augmentation:
                img = self.augmentation(img)
            
            # 应用转换
            if self.transform:
                img = self.transform(img)
            
            # 提取位置和类别标签
            location = torch.tensor(label[:4], dtype=torch.float32)
            class_id = torch.tensor(label[4], dtype=torch.long)
            
            return img, location, class_id
        
        except Exception as e:
            logger.error(f"加载图像失败: {img_path}, 错误: {e}")
            # 返回空数据或跳过
            return None

# 图像定位模型
class LocalizationModel(nn.Module):
    """结合定位和分类的双任务模型"""
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        参数:
            num_classes: 分类任务类别数
            pretrained: 是否使用预训练权重
        """
        super(LocalizationModel, self).__init__()
        
        # 使用ResNet101作为骨干网络
        self.backbone = torchvision.models.resnet101(pretrained=pretrained)
        
        # 移除原始分类层
        in_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1]
        
        # 定位分支 - 回归边界框坐标
        self.loc_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)  # xmin, ymin, xmax, ymax
        )
        
        # 分类分支
        self.cls_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # 定位输出 (边界框坐标)
        location = self.loc_head(features)
        
        # 分类输出 (类别概率)
        class_logits = self.cls_head(features)
        
        return location, class_logits

# 训练器类
class Trainer:
    """模型训练和评估类"""
    
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.device
        
        # 损失函数
        self.loc_criterion = nn.MSELoss()
        self.cls_criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        
        # 学习率调度器
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.5
        )
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'loc_loss': [],
            'cls_loss': [],
            'cls_accuracy': [],
            'iou': []
        }
        
        # 将模型移至设备
        self.model.to(self.device)
    
    def train_epoch(self, epoch):
        """训练单个epoch"""
        self.model.train()
        total_loss = 0.0
        total_loc_loss = 0.0
        total_cls_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm显示进度条
        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch+1}/{self.config.epochs}")
        
        for images, locations, class_ids in pbar:
            # 移至设备
            images = images.to(self.device)
            locations = locations.to(self.device)
            class_ids = class_ids.to(self.device)
            
            # 前向传播
            loc_preds, cls_preds = self.model(images)
            
            # 计算损失
            loc_loss = self.loc_criterion(loc_preds, locations)
            cls_loss = self.cls_criterion(cls_preds, class_ids)
            loss = loc_loss + cls_loss
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_loc_loss += loc_loss.item()
            total_cls_loss += cls_loss.item()
            
            # 计算分类准确率
            _, predicted = cls_preds.max(1)
            total += class_ids.size(0)
            correct += predicted.eq(class_ids).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'loc_loss': f"{loc_loss.item():.4f}",
                'cls_loss': f"{cls_loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        avg_loc_loss = total_loc_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        # 记录历史
        self.history['train_loss'].append(avg_loss)
        self.history['loc_loss'].append(avg_loc_loss)
        self.history['cls_loss'].append(avg_cls_loss)
        self.history['cls_accuracy'].append(accuracy)
        
        logger.info(f"训练 Epoch {epoch+1}/{self.config.epochs} - "
                   f"Loss: {avg_loss:.4f} | "
                   f"Loc Loss: {avg_loc_loss:.4f} | "
                   f"Cls Loss: {avg_cls_loss:.4f} | "
                   f"准确率: {accuracy:.2f}%")
        
        return avg_loss
    
    def evaluate(self, epoch=None):
        """在测试集上评估模型"""
        self.model.eval()
        total_loss = 0.0
        total_loc_loss = 0.0
        total_cls_loss = 0.0
        correct = 0
        total = 0
        iou_sum = 0.0
        
        # 用于分类报告
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="测试" if epoch is None else f"测试 Epoch {epoch+1}")
            
            for images, locations, class_ids in pbar:
                images = images.to(self.device)
                locations = locations.to(self.device)
                class_ids = class_ids.to(self.device)
                
                # 前向传播
                loc_preds, cls_preds = self.model(images)
                
                # 计算损失
                loc_loss = self.loc_criterion(loc_preds, locations)
                cls_loss = self.cls_criterion(cls_preds, class_ids)
                loss = loc_loss + cls_loss
                
                # 统计损失
                total_loss += loss.item()
                total_loc_loss += loc_loss.item()
                total_cls_loss += cls_loss.item()
                
                # 计算分类准确率
                _, predicted = cls_preds.max(1)
                total += class_ids.size(0)
                correct += predicted.eq(class_ids).sum().item()
                
                # 计算IoU (交并比)
                batch_iou = self.calculate_iou(loc_preds.cpu(), locations.cpu())
                iou_sum += batch_iou.sum().item()
                
                # 收集预测和目标
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(class_ids.cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.*correct/total:.2f}%",
                    'iou': f"{batch_iou.mean():.4f}"
                })
        
        # 计算平均指标
        avg_loss = total_loss / len(self.test_loader)
        avg_loc_loss = total_loc_loss / len(self.test_loader)
        avg_cls_loss = total_cls_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        avg_iou = iou_sum / total
        
        # 记录历史
        if epoch is not None:
            self.history['test_loss'].append(avg_loss)
            self.history['iou'].append(avg_iou)
        
        # 打印分类报告
        class_names = [self.config.class_names[i] for i in range(self.config.num_classes)]
        logger.info("\n分类报告:")
        logger.info(classification_report(
            all_targets, all_preds, target_names=class_names, digits=4
        ))
        
        # 打印评估结果
        eval_msg = (f"测试结果 - "
                   f"Loss: {avg_loss:.4f} | "
                   f"Loc Loss: {avg_loc_loss:.4f} | "
                   f"Cls Loss: {avg_cls_loss:.4f} | "
                   f"准确率: {accuracy:.2f}% | "
                   f"平均IoU: {avg_iou:.4f}")
        
        logger.info(eval_msg)
        
        return avg_loss, accuracy, avg_iou
    
    def calculate_iou(self, pred_boxes, true_boxes):
        """
        计算预测框和真实框之间的IoU
        参数:
            pred_boxes: 预测的边界框 [N, 4] (xmin, ymin, xmax, ymax)
            true_boxes: 真实的边界框 [N, 4] (xmin, ymin, xmax, ymax)
        返回:
            iou: 每个样本的IoU值 [N]
        """
        # 确保坐标在[0,1]范围内
        pred_boxes = torch.clamp(pred_boxes, 0, 1)
        true_boxes = torch.clamp(true_boxes, 0, 1)
        
        # 计算交集区域
        inter_xmin = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
        inter_ymin = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
        inter_xmax = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
        inter_ymax = torch.min(pred_boxes[:, 3], true_boxes[:, 3])
        
        inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)
        
        # 计算并集区域
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
        union_area = pred_area + true_area - inter_area
        
        # 计算IoU
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    def save_model(self, epoch, best=False):
        """保存模型检查点"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': vars(self.config)
        }
        
        filename = "model_best.pth" if best else f"model_epoch_{epoch+1}.pth"
        save_path = os.path.join(self.config.output_dir, filename)
        
        torch.save(state, save_path)
        logger.info(f"模型保存至: {save_path}")
    
    def load_model(self, checkpoint_path):
        """加载模型检查点"""
        if not os.path.exists(checkpoint_path):
            logger.error(f"检查点文件不存在: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        logger.info(f"加载检查点: {checkpoint_path}, 训练轮数: {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(self):
        """完整训练过程"""
        start_epoch = 0
        
        # 恢复训练
        if self.config.resume:
            start_epoch = self.load_model(self.config.resume)
        
        best_iou = 0.0
        
        for epoch in range(start_epoch, self.config.epochs):
            # 训练一个epoch
            self.train_epoch(epoch)
            
            # 评估
            _, _, avg_iou = self.evaluate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存最佳模型
            if avg_iou > best_iou:
                best_iou = avg_iou
                self.save_model(epoch, best=True)
            
            # 保存当前模型
            if (epoch + 1) % 5 == 0:
                self.save_model(epoch)
        
        # 保存最终模型
        self.save_model(self.config.epochs - 1)
        
        # 保存训练历史
        self.save_training_history()
        
        logger.info("训练完成!")
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.config.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 绘制训练曲线
        self.plot_training_history()
    
    def plot_training_history(self):
        """绘制训练历史曲线"""
        plt.figure(figsize=(15, 10))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['test_loss'], label='测试损失')
        plt.title('训练和测试损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        
        # 位置和分类损失
        plt.subplot(2, 2, 2)
        plt.plot(self.history['loc_loss'], label='位置损失')
        plt.plot(self.history['cls_loss'], label='分类损失')
        plt.title('位置和分类损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        
        # 分类准确率
        plt.subplot(2, 2, 3)
        plt.plot(self.history['cls_accuracy'], label='分类准确率')
        plt.title('分类准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率 (%)')
        plt.ylim(0, 100)
        
        # IoU
        plt.subplot(2, 2, 4)
        plt.plot(self.history['iou'], label='IoU', color='green')
        plt.title('边界框IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.ylim(0, 1)
        
        # 保存图像
        plt.tight_layout()
        plot_path = os.path.join(self.config.output_dir, "training_history.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"训练曲线保存至: {plot_path}")

# 可视化工具
class Visualizer:
    """预测结果可视化工具"""
    
    def __init__(self, model, config, transform):
        self.model = model
        self.config = config
        self.transform = transform
        self.device = config.device
        self.model.to(self.device)
        self.model.eval()
    
    def visualize_predictions(self, data_loader, num_samples=6):
        """可视化预测结果"""
        # 获取一个批次的数据
        images, locations, class_ids = next(iter(data_loader))
        
        # 移至设备并预测
        images = images.to(self.device)
        with torch.no_grad():
            loc_preds, cls_preds = self.model(images)
        
        # 转换预测结果
        loc_preds = loc_preds.cpu()
        cls_preds = cls_preds.cpu()
        
        # 创建可视化
        plt.figure(figsize=(15, 10))
        
        for i in range(min(num_samples, len(images))):
            plt.subplot(2, 3, i+1)
            
            # 原始图像
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]
            
            # 真实边界框
            true_xmin, true_ymin, true_xmax, true_ymax = locations[i].numpy() * self.config.image_size
            true_rect = Rectangle(
                (true_xmin, true_ymin), 
                true_xmax - true_xmin, 
                true_ymax - true_ymin,
                fill=False, 
                color='green',
                linewidth=2
            )
            
            # 预测边界框
            pred_xmin, pred_ymin, pred_xmax, pred_ymax = loc_preds[i].numpy() * self.config.image_size
            pred_rect = Rectangle(
                (pred_xmin, pred_ymin), 
                pred_xmax - pred_xmin, 
                pred_ymax - pred_ymin,
                fill=False, 
                color='red',
                linewidth=2,
                linestyle='--'
            )
            
            # 真实和预测类别
            true_class = self.config.class_names[class_ids[i].item()]
            pred_class = self.config.class_names[cls_preds[i].argmax().item()]
            
            # 绘制
            plt.imshow(img)
            ax = plt.gca()
            ax.add_patch(true_rect)
            ax.add_patch(pred_rect)
            plt.title(f"真实: {true_class}\n预测: {pred_class}")
            plt.axis('off')
        
        # 添加图例
        plt.figlegend(
            handles=[Rectangle((0,0),1,1,color='green'), Rectangle((0,0),1,1,color='red', linestyle='--')],
            labels=['真实框', '预测框'],
            loc='lower center',
            ncol=2
        )
        
        # 保存图像
        plot_path = os.path.join(self.config.output_dir, "predictions_visualization.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"预测可视化保存至: {plot_path}")

# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    config = Config(args)
    
    # 创建数据转换
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 数据预处理
    preprocessor = DataPreprocessor(config)
    train_data, test_data = preprocessor.create_datasets()
    
    # 创建数据集
    train_dataset = OxfordDataset(train_data, transform=transform, augment=True)
    test_dataset = OxfordDataset(test_data, transform=transform)
    
    # 创建数据加载器
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = LocalizationModel(num_classes=config.num_classes)
    logger.info(f"模型创建完成, 总参数数: {sum(p.numel() for p in model.parameters())}")
    
    # 训练或评估
    if config.eval_only:
        # 仅评估模式
        trainer = Trainer(model, None, test_loader, config)
        if config.resume:
            trainer.load_model(config.resume)
        trainer.evaluate()
    else:
        # 训练模式
        trainer = Trainer(model, train_loader, test_loader, config)
        trainer.train()
    
    # 可视化预测结果
    if config.visualize:
        visualizer = Visualizer(model, config, transform)
        visualizer.visualize_predictions(test_loader)

if __name__ == "__main__":
    main()
