"""
智能课堂分析系统 - 配置文件
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 基础路径
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据存储路径
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# 视频处理配置
VIDEO_CONFIG = {
    "supported_formats": ["mp4", "avi", "mkv", "hevc", "h265"],
    "frame_rate": 25,  # fps
    "resolution": (1280, 720),  # 默认分辨率
    "chunk_size": 10 * 1024 * 1024,  # 10MB 分块上传大小
}

# 模型配置
MODEL_CONFIG = {
    # YOLO行为识别模型
    "behavior_model": {
        "model_path": os.getenv("BEHAVIOR_MODEL_PATH", "models/yolov8x.pt"),
        "confidence": 0.5,
        "device": os.getenv("DEVICE", "cpu"),  # 'cpu' 或 'cuda'
    },
    # CLIP多模态对齐模型
    "clip_model": {
        "model_name": os.getenv("CLIP_MODEL_NAME", "ViT-L/14"),
        "device": os.getenv("DEVICE", "cpu"),
    },
    # Whisper语音转写模型
    "whisper_model": {
        "model_name": os.getenv("WHISPER_MODEL_NAME", "large-v3"),
        "language": "zh",
    },
    # DeepSeek大语言模型
    "llm_model": {
        "model_name": os.getenv("LLM_MODEL_NAME", "deepseek-moe-16b"),
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "temperature": 0.7,
        "max_tokens": 2048,
    },
}

# 行为分类配置 (基于弗兰德斯互动分析系统)
BEHAVIOR_CATEGORIES = {
    "teacher": [
        "讲解",  # 教师讲解内容
        "提问",  # 教师向学生提问
        "回应",  # 教师回应学生
        "板书",  # 教师在黑板上书写
        "示范",  # 教师演示操作
        "巡视",  # 教师在教室中走动
        "组织活动",  # 教师组织课堂活动
    ],
    "student": [
        "举手",  # 学生举手
        "回答",  # 学生回答问题
        "提问",  # 学生向教师提问
        "讨论",  # 学生之间讨论
        "做笔记",  # 学生记笔记
        "低头",  # 学生低头（可能分心）
        "转头",  # 学生转头（可能分心）
        "操作",  # 学生操作设备/实验
        "站立",  # 学生站立
        "走动",  # 学生走动
    ],
    "interaction": [
        "师生互动",  # 教师与学生互动
        "生生互动",  # 学生之间互动
        "小组活动",  # 小组活动
        "全班活动",  # 全班活动
    ],
}

# API配置
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
    "debug": os.getenv("DEBUG", "False").lower() == "true",
    "workers": int(os.getenv("WORKERS", 4)),
}

# 存储配置
STORAGE_CONFIG = {
    "minio": {
        "endpoint": os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        "access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        "secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        "secure": os.getenv("MINIO_SECURE", "False").lower() == "true",
        "bucket_name": os.getenv("MINIO_BUCKET", "classroom-videos"),
    },
    "mongodb": {
        "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017/"),
        "db_name": os.getenv("MONGO_DB", "classroom_analytics"),
    },
    "redis": {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", 6379)),
        "db": int(os.getenv("REDIS_DB", 0)),
        "password": os.getenv("REDIS_PASSWORD", None),
    },
}

# 日志配置
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(BASE_DIR, "logs", "app.log"),
}

# 创建必要的目录
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True) 