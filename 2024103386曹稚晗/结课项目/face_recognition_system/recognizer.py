from .models.face_encoder import FaceEncoder
from .detector import FaceDetector
from config.settings import Config
import numpy as np
from ..utils.image_utils import align_face


class FaceRecognizer:
    """人脸识别核心类"""

    def __init__(self):
        self.encoder = FaceEncoder()
        self.detector = FaceDetector()
        self.known_face_names = self.encoder.known_face_names
        self.known_face_encodings = self.encoder.known_face_encodings

    def process_frame(self, frame, align=False, landmarks=False):
        """
        处理视频帧，识别人脸
        :param frame: BGR格式的视频帧
        :param align: 是否对齐人脸
        :param landmarks: 是否显示特征点
        :return: (processed_frame, face_names) 处理后的帧和识别结果
        """
        # 检测人脸
        if landmarks and self.detector.landmark_predictor:
            face_locations, face_landmarks = self.detector.detect_faces_with_landmarks(frame[:, :, ::-1])
            face_encodings = []

            # 对齐人脸并编码
            if align:
                for i, (loc, landmarks) in enumerate(zip(face_locations, face_landmarks)):
                    aligned = align_face(frame[:, :, ::-1], landmarks)
                    encodings, _ = self.encoder.encode_face(aligned, [loc])
                    if encodings:
                        face_encodings.append(encodings[0])
            else:
                encodings, _ = self.encoder.encode_face(frame[:, :, ::-1], face_locations)
                face_encodings = encodings
        else:
            face_locations, face_encodings = self.detector.detect_faces(frame)
            face_landmarks = []

        # 识别人脸
        face_names = []
        for face_encoding in face_encodings:
            name, distance = self.encoder.recognize_face(face_encoding)
            if name != "Unknown":
                name = f"{name} ({distance:.2f})"
            face_names.append(name)

        # 绘制结果
        processed_frame = self.detector.draw_face_boxes(frame.copy(), face_locations, face_names)

        # 绘制特征点
        if landmarks and face_landmarks:
            processed_frame = self.detector.draw_landmarks(processed_frame, face_landmarks)

        return processed_frame, face_names

    def add_new_face(self, image, name, align=False):
        """
        添加新人脸到系统
        :param image: RGB图像
        :param name: 人脸对应的姓名
        :param align: 是否对齐人脸
        :return: 是否添加成功
        """
        if align and self.detector.landmark_predictor:
            _, landmarks = self.detector.detect_faces_with_landmarks(image)
            if landmarks:
                image = align_face(image, landmarks[0])

        return self.encoder.add_new_face(image, name)

    def get_known_faces(self):
        """获取已知人脸列表"""
        return self.encoder.known_face_names

    def clear_known_faces(self):
        """清空已知人脸"""
        self.encoder.known_face_names = []
        self.encoder.known_face_encodings = []
        self.encoder.db.clear_faces()