import cv2
import numpy as np
import os
import logging
from typing import List, Dict


class Visualizer:
    def __init__(self, config: Dict):
        """
        初始化可视化器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger("object_detection")
        
        # 可视化配置
        self.font_scale = config.get("visualization", {}).get("font_scale", 0.5)
        self.thickness = config.get("visualization", {}).get("thickness", 2)
        self.colors = self._get_colors()
        
        self.logger.info("可视化器初始化完成")
    
    def _get_colors(self) -> Dict:
        """获取类别对应的颜色"""
        # 生成随机颜色
        np.random.seed(42)  # 固定种子，确保颜色一致性
        colors = {}
        
        # COCO类别颜色
        if self.config.get("dataset", "coco") == "coco":
            # 为COCO的80个类别生成颜色
            for i in range(81):  # 包括背景类
                colors[i] = tuple(np.random.randint(0, 256, 3).tolist())
        else:
            # 自定义类别颜色
            if "classes" in self.config:
                for i, _ in enumerate(self.config["classes"]):
                    colors[i] = tuple(np.random.randint(0, 256, 3).tolist())
        
        return colors
    
    def visualize(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像，numpy数组，BGR格式
            detections: 检测结果列表
            
        Returns:
            可视化后的图像，numpy数组，BGR格式
        """
        vis_image = image.copy()
        
        for detection in detections:
            class_id = detection["class_id"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            bbox = detection["bbox"]
            
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, bbox)
            
            # 获取颜色
            color = self.colors.get(class_id, (0, 255, 0))
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, self.thickness)
            
            # 准备标签文本
            label = f"{class_name}: {confidence:.2f}"
            
            # 获取文本尺寸
            (text_width, text_height), _ = cv2.getTextSize(label, 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                                          self.font_scale, 
                                                          self.thickness)
            
            # 绘制文本背景
            cv2.rectangle(vis_image, (x1, y1 - text_height - 5), 
                         (x1 + text_width, y1), color, -1)
            
            # 绘制文本
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 
                       (255, 255, 255), self.thickness)
        
        return vis_image    