import cv2
import numpy as np
from typing import List, Tuple, Dict


def resize_image(image: np.ndarray, max_dimension: int = None) -> np.ndarray:
    """
    按比例调整图像大小
    :param image: 输入图像
    :param max_dimension: 最大边长(保持比例)
    :return: 调整后的图像
    """
    if max_dimension is None:
        return image

    h, w = image.shape[:2]
    if max(h, w) <= max_dimension:
        return image

    scale = max_dimension / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def align_face(image: np.ndarray, landmarks: Dict) -> np.ndarray:
    """
    根据面部特征点对齐人脸
    :param image: RGB图像
    :param landmarks: 面部特征点字典
    :return: 对齐后的人脸图像
    """
    # 获取左右眼坐标
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']

    # 计算眼睛角度
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # 计算旋转中心
    eyes_center = (
        (left_eye[0] + right_eye[0]) // 2,
        (left_eye[1] + right_eye[1]) // 2
    )

    # 获取旋转矩阵并执行旋转
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return rotated


def normalize_face(image: np.ndarray) -> np.ndarray:
    """
    标准化人脸图像(直方图均衡化)
    :param image: 人脸图像
    :return: 标准化后的图像
    """
    if len(image.shape) == 3:
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # 直方图均衡化
    equalized = cv2.equalizeHist(gray)

    if len(image.shape) == 3:
        # 转换回RGB
        equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

    return equalized


def preprocess_face(image: np.ndarray) -> np.ndarray:
    """人脸图像预处理(调整大小+标准化)"""
    processed = resize_image(image, Config.MAX_DIMENSION)
    processed = normalize_face(processed)
    return processed