import os
from pathlib import Path


class Config:
    """基础配置类"""
    BASE_DIR = Path(__file__).resolve().parent.parent
    DEBUG = False
    TESTING = False

    # 数据库配置
    DATABASE_URI = 'sqlite:///' + str(BASE_DIR / 'data' / 'faces.db')

    # 路径配置
    KNOWN_FACES_DIR = BASE_DIR / 'data' / 'known_faces'
    UNKNOWN_FACES_DIR = BASE_DIR / 'data' / 'unknown_faces'
    LANDMARK_MODEL = BASE_DIR / 'models' / 'shape_predictor_68_face_landmarks.dat'

    # 模型参数
    FACE_DETECTION_MODEL = 'hog'  # 或 'cnn'
    TOLERANCE = 0.6
    NUM_JITTERS = 1
    UPSAMPLE = 1
    MAX_DIMENSION = 1024

    # 性能设置
    FRAME_SKIP = 5  # 每5帧处理一次
    MIN_FACE_SIZE = 64  # 最小人脸像素尺寸

    @staticmethod
    def ensure_directories_exist():
        """确保所需目录存在"""
        os.makedirs(Config.KNOWN_FACES_DIR, exist_ok=True)
        os.makedirs(Config.UNKNOWN_FACES_DIR, exist_ok=True)
        os.makedirs(Config.BASE_DIR / 'models', exist_ok=True)


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    FACE_DETECTION_MODEL = 'hog'


class ProductionConfig(Config):
    """生产环境配置"""
    FACE_DETECTION_MODEL = 'cnn'
    NUM_JITTERS = 3
    UPSAMPLE = 2
    FRAME_SKIP = 2