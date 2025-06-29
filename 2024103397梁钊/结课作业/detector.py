import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import os
import logging
from typing import List, Dict, Tuple


class ObjectDetector:
    def __init__(self, config: Dict):
        """
        初始化目标检测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger("object_detection")
        
        # 模型配置
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                  and config.get("use_gpu", True) else "cpu")
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.nms_threshold = config.get("nms_threshold", 0.5)
        
        # 加载预训练模型
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # 类别标签
        self.classes = self._get_classes()
        
        self.logger.info(f"检测器使用设备: {self.device}")
    
    def _load_model(self) -> nn.Module:
        """加载预训练的目标检测模型"""
        model_type = self.config.get("model_type", "fasterrcnn_resnet50_fpn")
        
        if model_type == "fasterrcnn_resnet50_fpn":
            # 加载预训练的Faster R-CNN模型
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            
            # 如果有自定义权重，则加载
            if "model_weights" in self.config and os.path.exists(self.config["model_weights"]):
                try:
                    model.load_state_dict(torch.load(self.config["model_weights"], 
                                                    map_location=self.device))
                    self.logger.info(f"已加载自定义模型权重: {self.config['model_weights']}")
                except Exception as e:
                    self.logger.error(f"加载模型权重失败: {e}")
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return model
    
    def _get_classes(self) -> List[str]:
        """获取类别标签列表"""
        # COCO数据集类别
        if self.config.get("dataset", "coco") == "coco":
            return [
                '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
        # 自定义类别
        elif "classes" in self.config:
            return self.config["classes"]
        else:
            return []
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        对输入图像进行目标检测
        
        Args:
            image: 输入图像，numpy数组，形状为(H, W, C)，BGR格式
        
        Returns:
            检测结果列表，每个元素是一个字典，包含以下键:
                - class_id: 类别ID
                - class_name: 类别名称
                - confidence: 置信度分数
                - bbox: 边界框坐标[x1, y1, x2, y2]
        """
        # 转换图像格式
        image_tensor = self._preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # 模型推理
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        # 处理预测结果
        detections = self._postprocess_predictions(predictions[0])
        
        return detections
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        图像预处理，将BGR格式的numpy数组转换为RGB格式的张量
        
        Args:
            image: 输入图像，numpy数组，形状为(H, W, C)，BGR格式
        
        Returns:
            预处理后的图像张量
        """
        # 转换BGR到RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR_TO_RGB)
        
        # 转换为张量并归一化
        image_tensor = F.to_tensor(image)
        
        return image_tensor
    
    def _postprocess_predictions(self, predictions: Dict) -> List[Dict]:
        """
        后处理模型预测结果，包括过滤低置信度预测和应用NMS
        
        Args:
            predictions: 模型预测结果字典，包含以下键:
                - boxes: 边界框坐标，形状为(N, 4)
                - labels: 类别标签，形状为(N,)
                - scores: 置信度分数，形状为(N,)
        
        Returns:
            后处理后的检测结果列表
        """
        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        
        # 过滤低置信度预测
        indices = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        
        # 应用NMS
        if len(boxes) > 0:
            indices = self._apply_nms(boxes, scores, self.nms_threshold)
            boxes = boxes[indices]
            labels = labels[indices]
            scores = scores[indices]
        
        # 构建检测结果列表
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            if label < len(self.classes):
                class_name = self.classes[label]
            else:
                class_name = f"unknown_{label}"
            
            detections.append({
                "class_id": int(label),
                "class_name": class_name,
                "confidence": float(score),
                "bbox": [float(coord) for coord in box]  # [x1, y1, x2, y2]
            })
        
        return detections
    
    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        应用非极大值抑制(NMS)过滤重叠边界框
        
        Args:
            boxes: 边界框坐标，形状为(N, 4)
            scores: 置信度分数，形状为(N,)
            threshold: NMS阈值
        
        Returns:
            保留的边界框索引
        """
        if boxes.size == 0:
            return np.empty((0,), dtype=np.int32)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=np.int32)    