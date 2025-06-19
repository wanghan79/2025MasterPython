"""
智能课堂分析系统 - 数据模型和架构定义
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, HttpUrl


class VideoUploadRequest(BaseModel):
    """视频上传请求模型"""
    title: str = Field(..., description="视频标题")
    description: Optional[str] = Field(None, description="视频描述")
    course_name: Optional[str] = Field(None, description="课程名称")
    teacher_name: Optional[str] = Field(None, description="教师姓名")
    grade_level: Optional[str] = Field(None, description="年级")
    subject: Optional[str] = Field(None, description="学科")
    recording_date: Optional[datetime] = Field(None, description="录制日期")
    tags: Optional[List[str]] = Field(None, description="标签")


class VideoUploadResponse(BaseModel):
    """视频上传响应模型"""
    video_id: str = Field(..., description="视频ID")
    upload_url: HttpUrl = Field(..., description="上传URL")
    status: str = Field("pending", description="处理状态")
    message: str = Field("等待上传", description="状态消息")


class VideoInfo(BaseModel):
    """视频信息模型"""
    video_id: str = Field(..., description="视频ID")
    title: str = Field(..., description="视频标题")
    description: Optional[str] = Field(None, description="视频描述")
    course_name: Optional[str] = Field(None, description="课程名称")
    teacher_name: Optional[str] = Field(None, description="教师姓名")
    grade_level: Optional[str] = Field(None, description="年级")
    subject: Optional[str] = Field(None, description="学科")
    recording_date: Optional[datetime] = Field(None, description="录制日期")
    upload_date: datetime = Field(..., description="上传日期")
    duration: Optional[float] = Field(None, description="视频时长(秒)")
    file_size: Optional[int] = Field(None, description="文件大小(字节)")
    file_format: Optional[str] = Field(None, description="文件格式")
    resolution: Optional[str] = Field(None, description="分辨率")
    frame_rate: Optional[float] = Field(None, description="帧率")
    status: str = Field("pending", description="处理状态")
    tags: Optional[List[str]] = Field(None, description="标签")
    storage_path: str = Field(..., description="存储路径")
    thumbnail_url: Optional[HttpUrl] = Field(None, description="缩略图URL")


class TranscriptSegment(BaseModel):
    """转录文本片段"""
    start_time: float = Field(..., description="开始时间(秒)")
    end_time: float = Field(..., description="结束时间(秒)")
    text: str = Field(..., description="转录文本")
    speaker: Optional[str] = Field(None, description="说话人")
    confidence: float = Field(..., description="置信度")


class Transcript(BaseModel):
    """完整转录文本"""
    video_id: str = Field(..., description="视频ID")
    language: str = Field("zh", description="语言")
    segments: List[TranscriptSegment] = Field(..., description="转录片段")
    full_text: str = Field(..., description="完整文本")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    model_version: str = Field(..., description="模型版本")


class BoundingBox(BaseModel):
    """目标检测边界框"""
    x1: float = Field(..., description="左上角x坐标")
    y1: float = Field(..., description="左上角y坐标")
    x2: float = Field(..., description="右下角x坐标")
    y2: float = Field(..., description="右下角y坐标")


class DetectedBehavior(BaseModel):
    """检测到的行为"""
    frame_id: int = Field(..., description="帧ID")
    timestamp: float = Field(..., description="时间戳(秒)")
    category: str = Field(..., description="行为类别")
    label: str = Field(..., description="行为标签")
    confidence: float = Field(..., description="置信度")
    bbox: BoundingBox = Field(..., description="边界框")
    track_id: Optional[int] = Field(None, description="跟踪ID")
    attributes: Optional[Dict[str, Any]] = Field(None, description="附加属性")


class BehaviorAnalysis(BaseModel):
    """行为分析结果"""
    video_id: str = Field(..., description="视频ID")
    behaviors: List[DetectedBehavior] = Field(..., description="检测到的行为")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    model_version: str = Field(..., description="模型版本")
    summary: Optional[Dict[str, Any]] = Field(None, description="行为统计摘要")


class ClipEmbedding(BaseModel):
    """CLIP嵌入向量"""
    frame_id: int = Field(..., description="帧ID")
    timestamp: float = Field(..., description="时间戳(秒)")
    visual_embedding: List[float] = Field(..., description="视觉嵌入向量")
    text_embedding: Optional[List[float]] = Field(None, description="文本嵌入向量")
    alignment_score: Optional[float] = Field(None, description="对齐分数")


class MultimodalAlignment(BaseModel):
    """多模态对齐结果"""
    video_id: str = Field(..., description="视频ID")
    embeddings: List[ClipEmbedding] = Field(..., description="CLIP嵌入向量")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    model_version: str = Field(..., description="模型版本")


class AnalysisQuery(BaseModel):
    """分析查询请求"""
    video_id: str = Field(..., description="视频ID")
    query: str = Field(..., description="查询文本")
    time_range: Optional[List[float]] = Field(None, description="时间范围[开始,结束]")
    behavior_filter: Optional[List[str]] = Field(None, description="行为过滤")
    max_results: Optional[int] = Field(10, description="最大结果数")


class AnalysisResult(BaseModel):
    """分析结果响应"""
    query: str = Field(..., description="查询文本")
    response: str = Field(..., description="分析响应")
    video_id: str = Field(..., description="视频ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    relevant_segments: Optional[List[Dict[str, Any]]] = Field(None, description="相关片段")
    visualization_data: Optional[Dict[str, Any]] = Field(None, description="可视化数据")


class ProcessingStatus(BaseModel):
    """处理状态"""
    video_id: str = Field(..., description="视频ID")
    status: str = Field(..., description="状态")
    progress: float = Field(0.0, description="进度(0-1)")
    current_stage: str = Field(..., description="当前阶段")
    message: Optional[str] = Field(None, description="状态消息")
    error: Optional[str] = Field(None, description="错误信息")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")


class VisualizationRequest(BaseModel):
    """可视化请求"""
    video_id: str = Field(..., description="视频ID")
    visualization_type: str = Field(..., description="可视化类型")
    time_range: Optional[List[float]] = Field(None, description="时间范围[开始,结束]")
    behavior_filter: Optional[List[str]] = Field(None, description="行为过滤")
    aggregation: Optional[str] = Field("time", description="聚合方式")
    resolution: Optional[str] = Field("medium", description="分辨率")


class VisualizationResponse(BaseModel):
    """可视化响应"""
    video_id: str = Field(..., description="视频ID")
    visualization_type: str = Field(..., description="可视化类型")
    data: Dict[str, Any] = Field(..., description="可视化数据")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间") 