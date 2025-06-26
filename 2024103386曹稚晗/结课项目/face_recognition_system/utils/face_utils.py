import os
import cv2
import numpy as np
from pathlib import Path
from config.settings import Config
from datetime import datetime


def load_image_file(file, mode='RGB'):
    """
    加载图像文件
    :param file: 图像文件路径
    :param mode: 返回图像模式 ('RGB' 或 'BGR')
    :return: 图像(numpy数组)
    """
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Image file not found: {file}")

    image = cv2.imread(file)
    if image is None:
        raise ValueError(f"Could not read image file: {file}")

    return image if mode.upper() == 'BGR' else image[:, :, ::-1]


def save_face_image(image, name, output_dir=None):
    """
    保存人脸图像
    :param image: RGB图像
    :param name: 人脸名称
    :param output_dir: 输出目录
    :return: 保存的文件路径
    """
    output_dir = output_dir or Config.KNOWN_FACES_DIR
    os.makedirs(output_dir, exist_ok=True)

    # 生成唯一文件名
    base_name = name.replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{base_name}_{timestamp}.jpg"
    file_path = Path(output_dir) / file_name

    # 保存图像(BGR格式)
    cv2.imwrite(str(file_path), image[:, :, ::-1])
    return file_path


def extract_faces(image, face_locations):
    """
    从图像中提取人脸区域
    :param image: RGB图像
    :param face_locations: 人脸位置列表
    :return: 人脸图像列表
    """
    faces = []
    for (top, right, bottom, left) in face_locations:
        face = image[top:bottom, left:right]
        faces.append(face)
    return faces


def calculate_face_sizes(face_locations):
    """计算人脸大小"""
    return [(bottom - top, right - left) for (top, right, bottom, left) in face_locations]