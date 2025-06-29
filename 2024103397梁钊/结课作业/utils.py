import yaml
import logging
import os
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Any


def load_config(config_path: str) -> Dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


def setup_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件目录
    
    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 清除已有处理器
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 创建文件处理器（如果指定了日志目录）
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # 添加文件处理器到日志记录器
        logger.addHandler(fh)
    
    # 创建格式化器并添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    if log_dir:
        fh.setFormatter(formatter)
    
    # 添加控制台处理器到日志记录器
    logger.addHandler(ch)
    
    return logger


def save_results(results: List[Dict], output_path: str) -> None:
    """
    保存检测结果到JSON文件
    
    Args:
        results: 检测结果列表
        output_path: 输出文件路径
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"保存结果失败: {e}")


def read_image(image_path: str) -> np.ndarray:
    """
    读取图像
    
    Args:
        image_path: 图像路径
    
    Returns:
        图像数组，BGR格式
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        return image
    except Exception as e:
        print(f"读取图像失败: {e}")
        return None


def resize_image(image: np.ndarray, target_size: int = 800) -> np.ndarray:
    """
    调整图像大小
    
    Args:
        image: 输入图像
        target_size: 目标尺寸
    
    Returns:
        调整大小后的图像
    """
    height, width = image.shape[:2]
    
    # 计算调整比例
    if max(height, width) > target_size:
        scale = target_size / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    return image


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个边界框的IoU（交并比）
    
    Args:
        box1: 第一个边界框，格式为[x1, y1, x2, y2]
        box2: 第二个边界框，格式为[x1, y1, x2, y2]
    
    Returns:
        IoU值
    """
    # 获取边界框坐标
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集区域坐标
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # 计算交集面积
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 计算并集面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou    