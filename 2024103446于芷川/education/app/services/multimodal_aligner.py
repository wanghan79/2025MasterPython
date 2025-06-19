"""
智能课堂分析系统 - 多模态对齐服务
基于CLIP模型实现视觉-文本跨模态关联
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image
from loguru import logger

from config.settings import MODEL_CONFIG, PROCESSED_DIR


class MultimodalAligner:
    """基于CLIP的多模态对齐服务"""
    
    def __init__(self):
        """初始化多模态对齐器"""
        self.config = MODEL_CONFIG["clip_model"]
        self.model_name = self.config["model_name"]
        self.device = self.config["device"]
        
        # 加载模型
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.load_model()
        
        # 处理队列
        self.processing_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        
        logger.info("多模态对齐服务已初始化")
    
    def load_model(self) -> None:
        """加载CLIP模型"""
        try:
            logger.info(f"正在加载CLIP模型: {self.model_name}")
            
            # 设置设备
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA不可用，使用CPU进行推理")
                self.device = "cpu"
            
            # 加载模型
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, 
                device=torch.device(self.device)
            )
            
            # 加载分词器
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            logger.info(f"CLIP模型加载成功，使用设备: {self.device}")
        
        except Exception as e:
            logger.error(f"加载CLIP模型失败: {str(e)}")
            raise
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        编码图像为特征向量
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像特征向量
        """
        try:
            # 确保模型已加载
            if self.model is None:
                self.load_model()
            
            # 读取图像
            image = Image.open(image_path).convert("RGB")
            
            # 预处理图像
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # 编码图像
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 转换为numpy数组
            return image_features.cpu().numpy()[0]
        
        except Exception as e:
            logger.error(f"编码图像失败: {str(e)}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        编码文本为特征向量
        
        Args:
            text: 文本内容
            
        Returns:
            文本特征向量
        """
        try:
            # 确保模型已加载
            if self.model is None or self.tokenizer is None:
                self.load_model()
            
            # 分词
            text_tokens = self.tokenizer([text]).to(self.device)
            
            # 编码文本
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 转换为numpy数组
            return text_features.cpu().numpy()[0]
        
        except Exception as e:
            logger.error(f"编码文本失败: {str(e)}")
            raise
    
    def compute_similarity(self, image_features: np.ndarray, text_features: np.ndarray) -> float:
        """
        计算图像和文本特征的相似度
        
        Args:
            image_features: 图像特征向量
            text_features: 文本特征向量
            
        Returns:
            相似度分数 (0-1)
        """
        # 计算余弦相似度
        similarity = np.dot(image_features, text_features)
        
        # 将相似度范围从[-1, 1]映射到[0, 1]
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def align_frame_with_transcript(self, frame_path: str, transcript_segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        将视频帧与转录文本对齐
        
        Args:
            frame_path: 帧文件路径
            transcript_segment: 转录文本片段
            
        Returns:
            对齐结果
        """
        try:
            # 提取帧信息
            frame_filename = os.path.basename(frame_path)
            frame_parts = frame_filename.split("_")
            
            if len(frame_parts) >= 3:
                frame_id = int(frame_parts[1])
                timestamp = float(frame_parts[2].split(".")[0])
            else:
                # 如果文件名格式不符合预期，使用默认值
                frame_id = 0
                timestamp = 0.0
            
            # 编码图像
            visual_embedding = self.encode_image(frame_path)
            
            # 编码文本
            text = transcript_segment["text"]
            text_embedding = self.encode_text(text)
            
            # 计算相似度
            alignment_score = self.compute_similarity(visual_embedding, text_embedding)
            
            # 创建对齐结果
            alignment = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "visual_embedding": visual_embedding.tolist(),
                "text_embedding": text_embedding.tolist(),
                "alignment_score": alignment_score,
                "transcript_segment": transcript_segment
            }
            
            return alignment
        
        except Exception as e:
            logger.error(f"对齐帧与转录文本失败: {str(e)}")
            raise
    
    def process_video_frames(self, video_id: str, frame_paths: List[str], 
                            transcript_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理视频的所有帧与转录文本的对齐
        
        Args:
            video_id: 视频ID
            frame_paths: 帧文件路径列表
            transcript_segments: 转录文本片段列表
            
        Returns:
            处理结果
        """
        try:
            logger.info(f"开始处理视频 {video_id} 的 {len(frame_paths)} 帧与 {len(transcript_segments)} 个转录片段的对齐")
            
            # 初始化结果
            all_alignments = []
            
            # 为每个帧找到对应的转录片段
            for i, frame_path in enumerate(frame_paths):
                # 提取帧时间戳
                frame_filename = os.path.basename(frame_path)
                frame_parts = frame_filename.split("_")
                
                if len(frame_parts) >= 3:
                    frame_id = int(frame_parts[1])
                    timestamp = float(frame_parts[2].split(".")[0])
                else:
                    # 如果文件名格式不符合预期，使用索引作为帧ID和时间戳
                    frame_id = i
                    timestamp = i / 25.0  # 假设25fps
                
                # 找到对应的转录片段
                matching_segments = []
                for segment in transcript_segments:
                    if segment["start_time"] <= timestamp <= segment["end_time"]:
                        matching_segments.append(segment)
                
                # 如果没有匹配的片段，跳过
                if not matching_segments:
                    continue
                
                # 对于每个匹配的片段，计算对齐
                for segment in matching_segments:
                    alignment = self.align_frame_with_transcript(frame_path, segment)
                    all_alignments.append(alignment)
                
                # 每100帧记录一次进度
                if (i + 1) % 100 == 0 or i == len(frame_paths) - 1:
                    logger.info(f"已处理 {i+1}/{len(frame_paths)} 帧 ({(i+1)/len(frame_paths)*100:.1f}%)")
            
            # 创建结果
            result = {
                "video_id": video_id,
                "embeddings": all_alignments,
                "model_version": self.model_name
            }
            
            logger.info(f"视频 {video_id} 多模态对齐完成，共生成 {len(all_alignments)} 个对齐结果")
            
            return result
        
        except Exception as e:
            logger.error(f"处理视频帧与转录文本对齐失败: {str(e)}")
            raise
    
    def queue_multimodal_alignment(self, video_id: str, frame_paths: List[str], 
                                  transcript_segments: List[Dict[str, Any]], 
                                  callback=None) -> None:
        """
        将多模态对齐任务加入队列
        
        Args:
            video_id: 视频ID
            frame_paths: 帧文件路径列表
            transcript_segments: 转录文本片段列表
            callback: 处理完成后的回调函数
        """
        self.processing_queue.put({
            "video_id": video_id,
            "frame_paths": frame_paths,
            "transcript_segments": transcript_segments,
            "callback": callback
        })
        logger.info(f"视频 {video_id} 已加入多模态对齐队列")
    
    def _process_queue(self) -> None:
        """处理队列中的多模态对齐任务"""
        while True:
            try:
                # 从队列获取任务
                task = self.processing_queue.get()
                video_id = task["video_id"]
                frame_paths = task["frame_paths"]
                transcript_segments = task["transcript_segments"]
                callback = task["callback"]
                
                logger.info(f"开始处理视频 {video_id} 的多模态对齐任务")
                
                try:
                    # 处理视频帧与转录文本对齐
                    result = self.process_video_frames(video_id, frame_paths, transcript_segments)
                    
                    # 保存结果
                    result_dir = os.path.join(PROCESSED_DIR, video_id, "alignment")
                    os.makedirs(result_dir, exist_ok=True)
                    
                    # 调用回调函数
                    if callback:
                        callback(result)
                    
                    logger.info(f"视频 {video_id} 多模态对齐任务完成")
                
                except Exception as e:
                    logger.error(f"处理视频 {video_id} 多模态对齐任务失败: {str(e)}")
                    
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
                logger.error(f"多模态对齐队列出错: {str(e)}")
                time.sleep(1)  # 避免CPU占用过高 