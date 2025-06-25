"""
智能课堂分析系统 - 行为识别服务
基于YOLOv8实现课堂行为检测
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from loguru import logger

from config.settings import MODEL_CONFIG, BEHAVIOR_CATEGORIES, PROCESSED_DIR


class BehaviorDetector:
    """基于YOLOv8的课堂行为检测服务"""
    
    def __init__(self):
        """初始化行为检测器"""
        self.config = MODEL_CONFIG["behavior_model"]
        self.model_path = self.config["model_path"]
        self.confidence = self.config["confidence"]
        self.device = self.config["device"]
        
        # 行为类别
        self.categories = BEHAVIOR_CATEGORIES
        self.all_behaviors = []
        for category, behaviors in self.categories.items():
            self.all_behaviors.extend(behaviors)
        
        # 加载模型
        self.model = None
        self.load_model()
        
        # 处理队列
        self.processing_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        
        logger.info("行为检测服务已初始化")
    
    def load_model(self) -> None:
        """加载YOLO模型"""
        try:
            logger.info(f"正在加载行为检测模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 设置设备
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA不可用，使用CPU进行推理")
                self.device = "cpu"
            
            logger.info(f"行为检测模型加载成功，使用设备: {self.device}")
        
        except Exception as e:
            logger.error(f"加载行为检测模型失败: {str(e)}")
            raise
    
    def detect_behaviors(self, image_path: str) -> List[Dict[str, Any]]:
        """
        检测单张图像中的行为
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            检测到的行为列表
        """
        try:
            # 确保模型已加载
            if self.model is None:
                self.load_model()
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 执行检测
            results = self.model(image, conf=self.confidence, device=self.device)
            
            # 解析结果
            detections = []
            for result in results:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # 获取边界框
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 获取置信度
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # 获取类别
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]
                    
                    # 确定行为类别
                    behavior_category = None
                    for category, behaviors in self.categories.items():
                        if class_name in behaviors:
                            behavior_category = category
                            break
                    
                    # 如果不在预定义的行为中，跳过
                    if behavior_category is None:
                        continue
                    
                    # 创建检测结果
                    detection = {
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        },
                        "confidence": confidence,
                        "category": behavior_category,
                        "label": class_name,
                        "track_id": None  # 单帧检测没有跟踪ID
                    }
                    
                    detections.append(detection)
            
            return detections
        
        except Exception as e:
            logger.error(f"行为检测失败: {str(e)}")
            raise
    
    def process_video_frames(self, video_id: str, frame_paths: List[str]) -> Dict[str, Any]:
        """
        处理视频的所有帧
        
        Args:
            video_id: 视频ID
            frame_paths: 帧文件路径列表
            
        Returns:
            处理结果
        """
        try:
            logger.info(f"开始处理视频 {video_id} 的 {len(frame_paths)} 帧")
            
            # 初始化结果
            all_behaviors = []
            behavior_counts = {category: {behavior: 0 for behavior in behaviors} 
                              for category, behaviors in self.categories.items()}
            
            # 处理每一帧
            for i, frame_path in enumerate(frame_paths):
                # 提取帧信息
                frame_filename = os.path.basename(frame_path)
                frame_parts = frame_filename.split("_")
                
                if len(frame_parts) >= 3:
                    frame_id = int(frame_parts[1])
                    timestamp = float(frame_parts[2].split(".")[0])
                else:
                    # 如果文件名格式不符合预期，使用索引作为帧ID和时间戳
                    frame_id = i
                    timestamp = i / 25.0  # 假设25fps
                
                # 检测行为
                detections = self.detect_behaviors(frame_path)
                
                # 添加帧信息
                for detection in detections:
                    behavior = {
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "category": detection["category"],
                        "label": detection["label"],
                        "confidence": detection["confidence"],
                        "bbox": detection["bbox"],
                        "track_id": detection["track_id"]
                    }
                    
                    all_behaviors.append(behavior)
                    
                    # 更新计数
                    behavior_counts[detection["category"]][detection["label"]] += 1
                
                # 每100帧记录一次进度
                if (i + 1) % 100 == 0 or i == len(frame_paths) - 1:
                    logger.info(f"已处理 {i+1}/{len(frame_paths)} 帧 ({(i+1)/len(frame_paths)*100:.1f}%)")
            
            # 计算总行为数
            total_behaviors = sum(sum(counts.values()) for counts in behavior_counts.values())
            
            # 计算每个类别的百分比
            behavior_percentages = {}
            for category, behaviors in behavior_counts.items():
                category_total = sum(behaviors.values())
                behavior_percentages[category] = {
                    behavior: count / total_behaviors * 100 if total_behaviors > 0 else 0
                    for behavior, count in behaviors.items()
                }
            
            # 创建结果摘要
            summary = {
                "total_frames": len(frame_paths),
                "total_behaviors": total_behaviors,
                "behavior_counts": behavior_counts,
                "behavior_percentages": behavior_percentages
            }
            
            # 保存结果
            result = {
                "video_id": video_id,
                "behaviors": all_behaviors,
                "summary": summary,
                "model_version": os.path.basename(self.model_path)
            }
            
            logger.info(f"视频 {video_id} 行为检测完成，共检测到 {total_behaviors} 个行为")
            
            return result
        
        except Exception as e:
            logger.error(f"处理视频帧失败: {str(e)}")
            raise
    
    def queue_behavior_detection(self, video_id: str, frame_paths: List[str], 
                                callback=None) -> None:
        """
        将行为检测任务加入队列
        
        Args:
            video_id: 视频ID
            frame_paths: 帧文件路径列表
            callback: 处理完成后的回调函数
        """
        self.processing_queue.put({
            "video_id": video_id,
            "frame_paths": frame_paths,
            "callback": callback
        })
        logger.info(f"视频 {video_id} 已加入行为检测队列")
    
    def _process_queue(self) -> None:
        """处理队列中的行为检测任务"""
        while True:
            try:
                # 从队列获取任务
                task = self.processing_queue.get()
                video_id = task["video_id"]
                frame_paths = task["frame_paths"]
                callback = task["callback"]
                
                logger.info(f"开始处理视频 {video_id} 的行为检测任务")
                
                try:
                    # 处理视频帧
                    result = self.process_video_frames(video_id, frame_paths)
                    
                    # 保存结果
                    result_dir = os.path.join(PROCESSED_DIR, video_id, "analysis")
                    os.makedirs(result_dir, exist_ok=True)
                    
                    # 调用回调函数
                    if callback:
                        callback(result)
                    
                    logger.info(f"视频 {video_id} 行为检测任务完成")
                
                except Exception as e:
                    logger.error(f"处理视频 {video_id} 行为检测任务失败: {str(e)}")
                    
                    # 处理失败结果
                    result = {
                        "video_id": video_id,
                        "status": "error",
                        "error": str(e)
                    }
                    
                    # 调用回调函数
                    if callback:
                        callback(result)
                
                finally:
                    # 标记任务完成
                    self.processing_queue.task_done()
            
            except Exception as e:
                logger.error(f"行为检测队列出错: {str(e)}")
                time.sleep(1)  # 避免CPU占用过高 