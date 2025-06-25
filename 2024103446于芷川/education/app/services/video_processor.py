"""
智能课堂分析系统 - 视频处理服务
"""

import os
import uuid
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue

import cv2
import numpy as np
import ffmpeg
from loguru import logger

from config.settings import VIDEO_CONFIG, UPLOAD_DIR, PROCESSED_DIR


class VideoProcessor:
    """视频处理服务，负责视频的预处理、分帧和基本信息提取"""
    
    def __init__(self):
        """初始化视频处理器"""
        self.supported_formats = VIDEO_CONFIG["supported_formats"]
        self.default_frame_rate = VIDEO_CONFIG["frame_rate"]
        self.default_resolution = VIDEO_CONFIG["resolution"]
        self.chunk_size = VIDEO_CONFIG["chunk_size"]
        self.processing_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        logger.info("视频处理服务已初始化")
    
    def generate_video_id(self) -> str:
        """生成唯一的视频ID"""
        return str(uuid.uuid4())
    
    def get_upload_path(self, video_id: str, file_extension: str) -> str:
        """获取上传文件的存储路径"""
        return os.path.join(UPLOAD_DIR, f"{video_id}.{file_extension}")
    
    def get_processed_path(self, video_id: str, file_extension: str) -> str:
        """获取处理后文件的存储路径"""
        return os.path.join(PROCESSED_DIR, f"{video_id}.{file_extension}")
    
    def validate_video_format(self, file_extension: str) -> bool:
        """验证视频格式是否支持"""
        return file_extension.lower() in self.supported_formats
    
    def extract_video_info(self, video_path: str) -> Dict[str, Any]:
        """提取视频基本信息"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] 
                                if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] 
                                if stream['codec_type'] == 'audio'), None)
            
            if not video_stream:
                raise ValueError("无法找到视频流")
            
            # 提取视频信息
            duration = float(probe['format']['duration'])
            file_size = int(probe['format']['size'])
            
            # 视频流信息
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            frame_rate = eval(video_stream['r_frame_rate'])  # 可能是分数形式，如"30000/1001"
            codec = video_stream['codec_name']
            
            # 音频信息
            has_audio = audio_stream is not None
            audio_codec = audio_stream['codec_name'] if has_audio else None
            
            return {
                "duration": duration,
                "file_size": file_size,
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "frame_rate": frame_rate,
                "video_codec": codec,
                "has_audio": has_audio,
                "audio_codec": audio_codec,
                "format": probe['format']['format_name']
            }
        except Exception as e:
            logger.error(f"提取视频信息失败: {str(e)}")
            raise
    
    def extract_frames(self, video_path: str, output_dir: str, 
                      frame_rate: Optional[float] = None, 
                      max_frames: Optional[int] = None) -> List[str]:
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            frame_rate: 提取帧率，如果为None则使用默认帧率
            max_frames: 最大提取帧数，如果为None则提取所有帧
            
        Returns:
            提取的帧文件路径列表
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取视频信息
            video_info = self.extract_video_info(video_path)
            original_frame_rate = video_info["frame_rate"]
            
            # 确定提取帧率
            extract_rate = frame_rate if frame_rate is not None else self.default_frame_rate
            
            # 计算帧间隔
            if extract_rate >= original_frame_rate:
                # 如果请求的帧率高于原始帧率，使用原始帧率
                interval = 1
                actual_rate = original_frame_rate
            else:
                # 否则，计算帧间隔
                interval = round(original_frame_rate / extract_rate)
                actual_rate = original_frame_rate / interval
            
            logger.info(f"从视频中提取帧，原始帧率: {original_frame_rate}，目标帧率: {extract_rate}，"
                       f"实际帧率: {actual_rate}，帧间隔: {interval}")
            
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / original_frame_rate
            
            # 确定最大帧数
            if max_frames is not None:
                max_frames = min(max_frames, frame_count // interval)
            else:
                max_frames = frame_count // interval
            
            logger.info(f"视频总帧数: {frame_count}，持续时间: {duration:.2f}秒，"
                       f"将提取最多 {max_frames} 帧")
            
            # 提取帧
            frame_paths = []
            frame_index = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames is not None and extracted_count >= max_frames):
                    break
                
                if frame_index % interval == 0:
                    # 计算时间戳
                    timestamp = frame_index / original_frame_rate
                    frame_path = os.path.join(output_dir, f"frame_{extracted_count:06d}_{timestamp:.3f}.jpg")
                    
                    # 保存帧
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    extracted_count += 1
                
                frame_index += 1
            
            cap.release()
            logger.info(f"成功从视频中提取了 {extracted_count} 帧")
            
            return frame_paths
        
        except Exception as e:
            logger.error(f"提取视频帧失败: {str(e)}")
            raise
    
    def generate_thumbnail(self, video_path: str, output_path: str, 
                          time_offset: float = 5.0, size: Tuple[int, int] = (320, 180)) -> str:
        """
        从视频生成缩略图
        
        Args:
            video_path: 视频文件路径
            output_path: 输出缩略图路径
            time_offset: 从视频开始的时间偏移(秒)
            size: 缩略图尺寸 (宽, 高)
            
        Returns:
            缩略图文件路径
        """
        try:
            # 获取视频信息
            video_info = self.extract_video_info(video_path)
            duration = video_info["duration"]
            
            # 确保时间偏移在视频范围内
            if time_offset >= duration:
                time_offset = duration / 3  # 使用视频三分之一处的帧
            
            # 使用ffmpeg提取帧并调整大小
            (
                ffmpeg
                .input(video_path, ss=time_offset)
                .filter('scale', size[0], size[1])
                .output(output_path, vframes=1)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            logger.info(f"成功生成缩略图: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"生成缩略图失败: {str(e)}")
            raise
    
    def convert_video_format(self, input_path: str, output_path: str, 
                            target_format: str = "mp4", 
                            video_codec: str = "libx264",
                            audio_codec: str = "aac",
                            bitrate: Optional[str] = None) -> str:
        """
        转换视频格式
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            target_format: 目标格式
            video_codec: 视频编解码器
            audio_codec: 音频编解码器
            bitrate: 比特率
            
        Returns:
            转换后的视频路径
        """
        try:
            # 基本转换配置
            stream = ffmpeg.input(input_path)
            
            # 视频流配置
            video_stream = stream.video
            if bitrate:
                video_stream = video_stream.filter('bitrate', bitrate)
            
            # 音频流配置
            audio_stream = stream.audio
            
            # 输出配置
            output_args = {
                'c:v': video_codec,
                'c:a': audio_codec,
            }
            
            # 执行转换
            (
                ffmpeg
                .output(video_stream, audio_stream, output_path, **output_args)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            logger.info(f"成功转换视频格式: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"转换视频格式失败: {str(e)}")
            raise
    
    def queue_video_processing(self, video_id: str, video_path: str, 
                              callback=None, **kwargs) -> None:
        """
        将视频处理任务加入队列
        
        Args:
            video_id: 视频ID
            video_path: 视频文件路径
            callback: 处理完成后的回调函数
            **kwargs: 其他处理参数
        """
        self.processing_queue.put({
            "video_id": video_id,
            "video_path": video_path,
            "callback": callback,
            "params": kwargs
        })
        logger.info(f"视频 {video_id} 已加入处理队列")
    
    def _process_queue(self) -> None:
        """处理队列中的视频处理任务"""
        while True:
            try:
                # 从队列获取任务
                task = self.processing_queue.get()
                video_id = task["video_id"]
                video_path = task["video_path"]
                callback = task["callback"]
                params = task["params"]
                
                logger.info(f"开始处理视频: {video_id}")
                
                try:
                    # 提取视频信息
                    video_info = self.extract_video_info(video_path)
                    
                    # 生成缩略图
                    thumbnail_dir = os.path.join(PROCESSED_DIR, video_id, "thumbnails")
                    os.makedirs(thumbnail_dir, exist_ok=True)
                    thumbnail_path = os.path.join(thumbnail_dir, "thumbnail.jpg")
                    self.generate_thumbnail(video_path, thumbnail_path)
                    
                    # 提取帧
                    if params.get("extract_frames", True):
                        frames_dir = os.path.join(PROCESSED_DIR, video_id, "frames")
                        frame_rate = params.get("frame_rate", self.default_frame_rate)
                        max_frames = params.get("max_frames", None)
                        frame_paths = self.extract_frames(
                            video_path, frames_dir, frame_rate, max_frames
                        )
                    else:
                        frame_paths = []
                    
                    # 转换格式(如果需要)
                    if params.get("convert_format", False):
                        target_format = params.get("target_format", "mp4")
                        video_codec = params.get("video_codec", "libx264")
                        audio_codec = params.get("audio_codec", "aac")
                        bitrate = params.get("bitrate", None)
                        
                        converted_path = os.path.join(
                            PROCESSED_DIR, f"{video_id}.{target_format}"
                        )
                        self.convert_video_format(
                            video_path, converted_path, 
                            target_format, video_codec, audio_codec, bitrate
                        )
                    else:
                        converted_path = None
                    
                    # 处理结果
                    result = {
                        "video_id": video_id,
                        "video_info": video_info,
                        "thumbnail_path": thumbnail_path,
                        "frame_paths": frame_paths,
                        "converted_path": converted_path,
                        "status": "success"
                    }
                    
                    logger.info(f"视频 {video_id} 处理完成")
                    
                    # 调用回调函数
                    if callback:
                        callback(result)
                
                except Exception as e:
                    logger.error(f"处理视频 {video_id} 失败: {str(e)}")
                    
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
                logger.error(f"视频处理队列出错: {str(e)}")
                time.sleep(1)  # 避免CPU占用过高 