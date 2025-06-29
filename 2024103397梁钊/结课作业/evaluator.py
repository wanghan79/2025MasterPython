import os
import json
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from utils import calculate_iou


class Evaluator:
    def __init__(self, config: Dict):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger("object_detection")
        
        # 评估配置
        self.iou_threshold = config.get("evaluation", {}).get("iou_threshold", 0.5)
        self.confidence_thresholds = np.linspace(0.0, 1.0, 101)
        
        # 获取类别列表
        if config.get("dataset", "coco") == "coco":
            # COCO类别（不包括背景）
            self.classes = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
        elif "classes" in config:
            # 自定义类别
            self.classes = config["classes"]
        else:
            self.classes = []
        
        self.logger.info("评估器初始化完成")
    
    def evaluate(self, results: List[Dict]) -> Dict:
        """
        评估模型性能
        
        Args:
            results: 检测结果列表
            
        Returns:
            评估指标字典
        """
        self.logger.info("开始评估模型性能")
        
        # 加载真实标注（如果有）
        ground_truth_dir = self.config.get("ground_truth_dir")
        if not ground_truth_dir or not os.path.exists(ground_truth_dir):
            self.logger.warning("未找到真实标注目录，无法进行评估")
            return {}
        
        # 读取真实标注
        ground_truth = self._load_ground_truth(ground_truth_dir)
        
        # 计算评估指标
        metrics = self._calculate_metrics(results, ground_truth)
        
        return metrics
    
    def _load_ground_truth(self, ground_truth_dir: str) -> Dict:
        """
        加载真实标注数据
        
        Args:
            ground_truth_dir: 真实标注目录
            
        Returns:
            真实标注字典
        """
        ground_truth = {}
        
        # 假设标注文件为JSON格式，每个文件对应一张图像
        for filename in os.listdir(ground_truth_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(ground_truth_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # 提取图像ID（假设为文件名的前缀）
                    image_id = os.path.splitext(filename)[0]
                    
                    # 提取标注信息
                    annotations = []
                    for ann in data.get('annotations', []):
                        # 确保标注包含必要的字段
                        if 'category_id' in ann and 'bbox' in ann:
                            # 转换边界框格式 [x, y, width, height] -> [x1, y1, x2, y2]
                            x, y, w, h = ann['bbox']
                            bbox = [x, y, x + w, y + h]
                            
                            annotations.append({
                                'class_id': ann['category_id'],
                                'bbox': bbox,
                                'iscrowd': ann.get('iscrowd', 0)
                            })
                    
                    ground_truth[image_id] = annotations
                except Exception as e:
                    self.logger.warning(f"加载标注文件 {filename} 失败: {e}")
        
        return ground_truth
    
    def _calculate_metrics(self, results: List[Dict], ground_truth: Dict) -> Dict:
        """
        计算评估指标
        
        Args:
            results: 检测结果列表
            ground_truth: 真实标注字典
            
        Returns:
            评估指标字典
        """
        # 初始化指标存储
        metrics = {
            'AP': {},  # 各类别的AP
            'mAP': 0.0,  # 平均AP
            'precision': {},  # 各类别的精度-召回曲线
            'recall': {},  # 各类别的精度-召回曲线
            'TP': {},  # 各类别的真阳性数
            'FP': {},  # 各类别的假阳性数
            'FN': {},  # 各类别的假阴性数
            'total_positives': {}  # 各类别的总阳性数
        }
        
        # 按类别收集检测结果和真实标注
        detections_by_class = {i: [] for i in range(len(self.classes))}
        gt_by_class = {i: [] for i in range(len(self.classes))}
        
        # 处理每个图像的检测结果
        for result in results:
            image_id = self._get_image_id(result['image_path'])
            detections = result['detections']
            
            # 获取该图像的真实标注
            gt_annotations = ground_truth.get(image_id, [])
            
            # 按类别收集检测结果
            for det in detections:
                class_id = det['class_id']
                if class_id < len(self.classes):
                    detections_by_class[class_id].append({
                        'bbox': det['bbox'],
                        'confidence': det['confidence'],
                        'image_id': image_id
                    })
            
            # 按类别收集真实标注
            for gt in gt_annotations:
                class_id = gt['class_id']
                if class_id < len(self.classes):
                    gt_by_class[class_id].append({
                        'bbox': gt['bbox'],
                        'image_id': image_id,
                        'detected': False  # 标记是否已被检测到
                    })
        
        # 计算每个类别的AP
        all_ap = []
        for class_id in range(len(self.classes)):
            class_name = self.classes[class_id]
            detections = detections_by_class[class_id]
            gt_instances = gt_by_class[class_id]
            
            # 跳过没有检测结果或真实标注的类别
            if len(detections) == 0 or len(gt_instances) == 0:
                metrics['AP'][class_name] = 0.0
                metrics['precision'][class_name] = np.zeros_like(self.confidence_thresholds)
                metrics['recall'][class_name] = np.zeros_like(self.confidence_thresholds)
                metrics['TP'][class_name] = np.zeros_like(self.confidence_thresholds)
                metrics['FP'][class_name] = np.zeros_like(self.confidence_thresholds)
                metrics['FN'][class_name] = len(gt_instances)
                metrics['total_positives'][class_name] = len(gt_instances)
                continue
            
            # 按置信度排序
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            
            # 初始化TP和FP数组
            tp = np.zeros(len(detections))
            fp = np.zeros(len(detections))
            
            # 总真实阳性数
            total_positives = len(gt_instances)
            
            # 遍历每个检测结果
            for d_idx, det in enumerate(detections):
                # 获取该图像的真实标注
                image_gt = [gt for gt in gt_instances if gt['image_id'] == det['image_id']]
                
                # 初始化最大IoU和匹配的真实标注索引
                max_iou = 0.0
                gt_match_idx = -1
                
                # 查找与当前检测框IoU最大的真实标注
                for gt_idx, gt in enumerate(image_gt):
                    if not gt['detected']:
                        iou = calculate_iou(det['bbox'], gt['bbox'])
                        if iou > max_iou:
                            max_iou = iou
                            gt_match_idx = gt_idx
                
                # 判断是TP还是FP
                if max_iou >= self.iou_threshold:
                    # 标记该真实标注已被检测到
                    image_gt[gt_match_idx]['detected'] = True
                    tp[d_idx] = 1
                else:
                    fp[d_idx] = 1
            
            # 计算累积TP和FP
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            
            # 计算精度和召回率
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / total_positives
            
            # 计算AP（使用11点插值法）
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                ap += p / 11.0
            
            # 存储结果
            metrics['AP'][class_name] = ap
            metrics['precision'][class_name] = precision
            metrics['recall'][class_name] = recall
            metrics['TP'][class_name] = cum_tp
            metrics['FP'][class_name] = cum_fp
            metrics['FN'][class_name] = total_positives - cum_tp[-1] if total_positives > 0 else 0
            metrics['total_positives'][class_name] = total_positives
            
            all_ap.append(ap)
        
        # 计算mAP
        if all_ap:
            metrics['mAP'] = np.mean(all_ap)
        
        return metrics
    
    def _get_image_id(self, image_path: str) -> str:
        """
        从图像路径获取图像ID
        
        Args:
            image_path: 图像路径
            
        Returns:
            图像ID
        """
        # 提取文件名（不包含扩展名）
        return os.path.splitext(os.path.basename(image_path))[0]
    
    def save_metrics(self, metrics: Dict, output_path: str) -> None:
        """
        保存评估指标到JSON文件
        
        Args:
            metrics: 评估指标字典
            output_path: 输出文件路径
        """
        try:
            # 转换numpy数组为列表以便保存为JSON
            metrics_to_save = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    metrics_to_save[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray):
                            metrics_to_save[key][sub_key] = sub_value.tolist()
                        else:
                            metrics_to_save[key][sub_key] = sub_value
                else:
                    metrics_to_save[key] = value
            
            with open(output_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)
            
            self.logger.info(f"评估指标已保存至: {output_path}")
        except Exception as e:
            self.logger.error(f"保存评估指标失败: {e}")
    
    def plot_metrics(self, metrics: Dict, output_dir: str) -> None:
        """
        绘制评估指标图表
        
        Args:
            metrics: 评估指标字典
            output_dir: 输出目录
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 绘制AP曲线
            if 'AP' in metrics:
                plt.figure(figsize=(12, 8))
                plt.bar(metrics['AP'].keys(), metrics['AP'].values())
                plt.xticks(rotation=90)
                plt.title('Average Precision (AP) per Class')
                plt.xlabel('Class')
                plt.ylabel('AP')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'ap_per_class.png'))
                plt.close()
            
            # 绘制mAP
            if 'mAP' in metrics:
                plt.figure(figsize=(6, 6))
                plt.bar(['mAP'], [metrics['mAP']])
                plt.ylim(0, 1)
                plt.title('Mean Average Precision (mAP)')
                plt.ylabel('mAP')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'map.png'))
                plt.close()
            
            # 绘制精度-召回曲线
            if 'precision' in metrics and 'recall' in metrics:
                plt.figure(figsize=(10, 10))
                for class_name in metrics['precision']:
                    precision = metrics['precision'][class_name]
                    recall = metrics['recall'][class_name]
                    plt.plot(recall, precision, label=class_name)
                
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend(loc='best')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
                plt.close()
            
            self.logger.info(f"评估图表已保存至: {output_dir}")
        except Exception as e:
            self.logger.error(f"绘制评估图表失败: {e}")    