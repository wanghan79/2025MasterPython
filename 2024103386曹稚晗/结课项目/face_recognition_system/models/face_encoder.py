import face_recognition
import numpy as np
import pickle
import os
from pathlib import Path
from config.settings import Config
from ..database import FaceDatabase
from ..utils.image_utils import resize_image


class FaceEncoder:
    """人脸编码和识别核心类"""

    def __init__(self, db=None):
        self.db = db or FaceDatabase()
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """从数据库加载已知人脸编码"""
        known_faces = self.db.get_all_faces()
        for face in known_faces:
            try:
                encoding = pickle.loads(face['encoding'])
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(face['name'])
            except Exception as e:
                print(f"Error loading face {face['name']}: {str(e)}")

    def encode_face(self, image, boxes=None):
        """
        对人脸图像进行编码
        :param image: RGB图像(numpy数组)
        :param boxes: 可选的人脸位置列表
        :return: (encodings, boxes) 人脸编码和位置列表
        """
        if boxes is None:
            boxes = face_recognition.face_locations(
                image,
                model=Config.FACE_DETECTION_MODEL,
                number_of_times_to_upsample=Config.UPSAMPLE
            )

        encodings = face_recognition.face_encodings(
            image,
            boxes,
            num_jitters=Config.NUM_JITTERS
        )
        return encodings, boxes

    def recognize_face(self, face_encoding):
        """
        识别人脸
        :param face_encoding: 要识别的人脸编码
        :return: (name, distance) 姓名和距离
        """
        if not self.known_face_encodings:
            return "Unknown", 1.0

        distances = face_recognition.face_distance(
            self.known_face_encodings,
            face_encoding
        )
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        if min_distance <= Config.TOLERANCE:
            return self.known_face_names[min_index], min_distance
        return "Unknown", min_distance

    def add_new_face(self, image, name, save_image=True):
        """
        添加新人脸到系统
        :param image: RGB图像
        :param name: 人脸对应的姓名
        :param save_image: 是否保存图像到文件
        :return: 是否添加成功
        """
        # 调整图像大小
        image = resize_image(image, Config.MAX_DIMENSION)

        encodings, boxes = self.encode_face(image)
        if not encodings:
            raise ValueError("No face found in the image")

        # 保存到数据库
        self.db.add_face(name, pickle.dumps(encodings[0]))

        # 更新内存中的列表
        self.known_face_encodings.append(encodings[0])
        self.known_face_names.append(name)

        # 保存图像到文件
        if save_image:
            from ..utils.face_utils import save_face_image
            save_face_image(image, name)

        return True

    def batch_add_faces(self, images, names):
        """批量添加人脸"""
        results = []
        for image, name in zip(images, names):
            try:
                self.add_new_face(image, name)
                results.append((name, True, ""))
            except Exception as e:
                results.append((name, False, str(e)))
        return results