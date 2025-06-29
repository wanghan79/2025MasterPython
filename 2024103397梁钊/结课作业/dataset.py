import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from utils import read_image, resize_image


class ImageDataset:
    def __init__(self, input_path: str, config: Dict):
        """
        初始化图像数据集
        
        Args:
            input_path: 输入图像路径或目录
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger("object_detection")
        
        # 获取图像路径列表
        self.image_paths = self._get_image_paths(input_path)
        self.logger.info(f"找到 {len(self.image_paths)} 张图像")
        
        # 预处理配置
        self.target_size = config.get("preprocessing", {}).get("target_size", 800)
        self.max_size = config.get("preprocessing", {}).get("max_size", 1333)
    
    def _get_image_paths(self, input_path: str) -> List[str]:
        """
        获取图像路径列表
        
        Args:
            input_path: 输入图像路径或目录
        
        Returns:
            图像路径列表
        """
        if os.path.isfile(input_path):
            # 单个图像文件
            return [input_path]
        elif os.path.isdir(input_path):
            # 图像目录
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            return [os.path.join(input_path, f) 
                   for f in os.listdir(input_path) 
                   if os.path.isfile(os.path.join(input_path, f)) 
                   and os.path.splitext(f)[1].lower() in image_extensions]
        else:
            self.logger.error(f"路径不存在: {input_path}")
            return []
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Optional[np.ndarray], str]:
        """
        获取数据集中的一项
        
        Args:
            idx: 索引
        
        Returns:
            元组(image, image_path)，如果图像读取失败则image为None
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("索引超出范围")
        
        image_path = self.image_paths[idx]
        image = read_image(image_path)
        
        if image is not None:
            # 调整图像大小
            if self.target_size > 0:
                image = resize_image(image, self.target_size)
        
        return image, image_path


class VideoDataset:
    def __init__(self, input_path: str, config: Dict):
        """
        初始化视频数据集
        
        Args:
            input_path: 输入视频路径
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger("object_detection")
        
        # 打开视频文件
        self.video_path = input_path
        self.video_capture = cv2.VideoCapture(input_path)
        
        if not self.video_capture.isOpened():
            self.logger.error(f"无法打开视频文件: {input_path}")
            raise ValueError(f"无法打开视频文件: {input_path}")
        
        # 获取视频信息
        self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.logger.info(f"视频信息: 帧数={self.frame_count}, FPS={self.fps:.2f}")
        
        # 预处理配置
        self.target_size = config.get("preprocessing", {}).get("target_size", 800)
        self.sample_rate = config.get("video", {}).get("sample_rate", 1)  # 采样率，每隔多少帧处理一次
        self.current_frame = 0
    
    def __len__(self) -> int:
        """返回数据集大小（总帧数）"""
        return (self.frame_count + self.sample_rate - 1) // self.sample_rate
    
    def __getitem__(self, idx: int) -> Tuple[Optional[np.ndarray], str]:
        """
        获取数据集中的一项（一帧）
        
        Args:
            idx: 索引
        
        Returns:
            元组(image, frame_info)，如果读取失败则image为None
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("索引超出范围")
        
        # 计算实际帧号
        frame_idx = idx * self.sample_rate
        
        # 设置视频捕获位置
        if frame_idx != self.current_frame:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.current_frame = frame_idx
        
        # 读取帧
        ret, frame = self.video_capture.read()
        self.current_frame += 1
        
        if not ret:
            self.logger.warning(f"无法读取帧: {frame_idx}")
            return None, f"frame_{frame_idx}"
        
        # 调整图像大小
        if self.target_size > 0:
            frame = resize_image(frame, self.target_size)
        
        return frame, f"frame_{frame_idx}"
    
    def release(self) -> None:
        """释放视频捕获资源"""
        if self.video_capture.isOpened():
            self.video_capture.release()    