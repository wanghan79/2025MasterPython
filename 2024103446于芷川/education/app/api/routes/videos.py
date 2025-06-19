"""
智能课堂分析系统 - 视频管理路由
"""

import os
import shutil
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse, FileResponse
from pydantic import HttpUrl
from loguru import logger

from app.models.schemas import VideoUploadRequest, VideoUploadResponse, VideoInfo, ProcessingStatus
from app.services.video_processor import VideoProcessor
from app.services.behavior_detector import BehaviorDetector
from app.services.multimodal_aligner import MultimodalAligner
from config.settings import UPLOAD_DIR, PROCESSED_DIR

# 创建路由
router = APIRouter()

# 服务实例
video_processor = VideoProcessor()
behavior_detector = BehaviorDetector()
multimodal_aligner = MultimodalAligner()

# 处理状态存储（实际应用中应使用数据库或Redis）
processing_status = {}

# 视频信息存储（实际应用中应使用数据库）
video_info_store = {}

@router.post("/upload", response_model=VideoUploadResponse, summary="上传视频")
async def upload_video(
    background_tasks: BackgroundTasks,
    title: str = Form(..., description="视频标题"),
    description: Optional[str] = Form(None, description="视频描述"),
    course_name: Optional[str] = Form(None, description="课程名称"),
    teacher_name: Optional[str] = Form(None, description="教师姓名"),
    grade_level: Optional[str] = Form(None, description="年级"),
    subject: Optional[str] = Form(None, description="学科"),
    tags: Optional[str] = Form(None, description="标签，用逗号分隔"),
    file: UploadFile = File(..., description="视频文件"),
):
    """
    上传视频文件并开始处理
    """
    try:
        # 生成视频ID
        video_id = video_processor.generate_video_id()
        
        # 获取文件扩展名
        file_extension = os.path.splitext(file.filename)[1][1:].lower()
        
        # 验证文件格式
        if not video_processor.validate_video_format(file_extension):
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的视频格式: {file_extension}，支持的格式: {', '.join(video_processor.supported_formats)}"
            )
        
        # 保存文件
        upload_path = video_processor.get_upload_path(video_id, file_extension)
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        
        # 创建文件
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 解析标签
        tag_list = tags.split(",") if tags else []
        
        # 创建视频信息
        video_info = VideoInfo(
            video_id=video_id,
            title=title,
            description=description,
            course_name=course_name,
            teacher_name=teacher_name,
            grade_level=grade_level,
            subject=subject,
            tags=tag_list,
            storage_path=upload_path,
            upload_date=None,  # 将在处理时设置
        )
        
        # 存储视频信息
        video_info_store[video_id] = video_info.dict()
        
        # 更新处理状态
        processing_status[video_id] = ProcessingStatus(
            video_id=video_id,
            status="uploading",
            progress=0.0,
            current_stage="上传",
            message="视频上传中",
        ).dict()
        
        # 在后台处理视频
        background_tasks.add_task(process_video, video_id, upload_path)
        
        # 返回响应
        return VideoUploadResponse(
            video_id=video_id,
            upload_url=f"/api/videos/{video_id}",
            status="uploading",
            message="视频上传成功，开始处理",
        )
    
    except Exception as e:
        logger.error(f"上传视频失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传视频失败: {str(e)}")

@router.get("/list", response_model=List[VideoInfo], summary="获取视频列表")
async def list_videos(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(10, ge=1, le=100, description="每页数量"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    tags: Optional[List[str]] = Query(None, description="标签过滤"),
):
    """
    获取视频列表，支持分页、搜索和标签过滤
    """
    try:
        # 过滤视频
        filtered_videos = []
        for video in video_info_store.values():
            # 搜索过滤
            if search and search.lower() not in video["title"].lower() and (
                not video["description"] or search.lower() not in video["description"].lower()
            ):
                continue
            
            # 标签过滤
            if tags and not all(tag in video.get("tags", []) for tag in tags):
                continue
            
            filtered_videos.append(video)
        
        # 分页
        start = (page - 1) * page_size
        end = start + page_size
        paginated_videos = filtered_videos[start:end]
        
        return paginated_videos
    
    except Exception as e:
        logger.error(f"获取视频列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取视频列表失败: {str(e)}")

@router.get("/{video_id}", response_model=VideoInfo, summary="获取视频信息")
async def get_video(video_id: str):
    """
    获取指定视频的详细信息
    """
    try:
        if video_id not in video_info_store:
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")
        
        return video_info_store[video_id]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取视频信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取视频信息失败: {str(e)}")

@router.get("/{video_id}/status", response_model=ProcessingStatus, summary="获取处理状态")
async def get_processing_status(video_id: str):
    """
    获取视频处理状态
    """
    try:
        if video_id not in processing_status:
            raise HTTPException(status_code=404, detail=f"视频处理状态不存在: {video_id}")
        
        return processing_status[video_id]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取处理状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取处理状态失败: {str(e)}")

@router.delete("/{video_id}", summary="删除视频")
async def delete_video(video_id: str):
    """
    删除指定视频及其相关数据
    """
    try:
        if video_id not in video_info_store:
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")
        
        # 获取视频信息
        video_info = video_info_store[video_id]
        
        # 删除上传文件
        upload_path = video_info["storage_path"]
        if os.path.exists(upload_path):
            os.remove(upload_path)
        
        # 删除处理目录
        processed_dir = os.path.join(PROCESSED_DIR, video_id)
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
        
        # 删除视频信息和处理状态
        del video_info_store[video_id]
        if video_id in processing_status:
            del processing_status[video_id]
        
        return JSONResponse(content={"message": f"视频 {video_id} 已删除"})
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除视频失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除视频失败: {str(e)}")

# 后台处理函数
async def process_video(video_id: str, video_path: str):
    """
    后台处理视频
    """
    try:
        # 更新处理状态
        processing_status[video_id] = ProcessingStatus(
            video_id=video_id,
            status="processing",
            progress=0.1,
            current_stage="预处理",
            message="开始处理视频",
        ).dict()
        
        # 处理视频
        def process_callback(result):
            if result["status"] == "success":
                # 更新视频信息
                if video_id in video_info_store:
                    video_info = video_info_store[video_id]
                    video_info.update({
                        "duration": result["video_info"]["duration"],
                        "file_size": result["video_info"]["file_size"],
                        "file_format": result["video_info"]["format"],
                        "resolution": result["video_info"]["resolution"],
                        "frame_rate": result["video_info"]["frame_rate"],
                        "thumbnail_url": f"/api/videos/{video_id}/thumbnail",
                        "status": "processed",
                    })
                
                # 更新处理状态
                processing_status[video_id] = ProcessingStatus(
                    video_id=video_id,
                    status="processed",
                    progress=1.0,
                    current_stage="完成",
                    message="视频处理完成",
                ).dict()
                
                # 开始行为检测
                frame_paths = result["frame_paths"]
                behavior_detector.queue_behavior_detection(video_id, frame_paths, behavior_callback)
            else:
                # 处理失败
                processing_status[video_id] = ProcessingStatus(
                    video_id=video_id,
                    status="error",
                    progress=0.0,
                    current_stage="错误",
                    message="视频处理失败",
                    error=result["error"],
                ).dict()
        
        # 行为检测回调
        def behavior_callback(result):
            if "status" in result and result["status"] == "error":
                # 处理失败
                processing_status[video_id] = ProcessingStatus(
                    video_id=video_id,
                    status="error",
                    progress=0.0,
                    current_stage="错误",
                    message="行为检测失败",
                    error=result["error"],
                ).dict()
            else:
                # 行为检测成功
                processing_status[video_id] = ProcessingStatus(
                    video_id=video_id,
                    status="analyzed",
                    progress=1.0,
                    current_stage="完成",
                    message="行为检测完成",
                ).dict()
        
        # 将视频处理任务加入队列
        video_processor.queue_video_processing(
            video_id, 
            video_path, 
            process_callback,
            extract_frames=True,
            frame_rate=25,
            convert_format=True,
            target_format="mp4",
        )
        
    except Exception as e:
        logger.error(f"处理视频失败: {str(e)}")
        # 更新处理状态
        processing_status[video_id] = ProcessingStatus(
            video_id=video_id,
            status="error",
            progress=0.0,
            current_stage="错误",
            message="处理视频失败",
            error=str(e),
        ).dict()

@router.get("/{video_id}/thumbnail", summary="获取视频缩略图")
async def get_thumbnail(video_id: str):
    """
    获取视频缩略图
    """
    try:
        if video_id not in video_info_store:
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")
        
        # 缩略图路径
        thumbnail_path = os.path.join(PROCESSED_DIR, video_id, "thumbnails", "thumbnail.jpg")
        
        if not os.path.exists(thumbnail_path):
            raise HTTPException(status_code=404, detail=f"缩略图不存在")
        
        # 返回图片文件
        return FileResponse(thumbnail_path, media_type="image/jpeg")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取缩略图失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取缩略图失败: {str(e)}") 