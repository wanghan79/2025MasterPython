import cv2
import face_recognition
import dlib
import numpy as np
from typing import List, Dict, Tuple
from config.settings import Config


class FaceDetector:
    """人脸检测和特征点检测类"""

    def __init__(self):
        self.frame_count = 0
        self.landmark_predictor = None
        self._init_landmark_detector()

    def _init_landmark_detector(self):
        """初始化面部特征点检测器"""
        if Config.LANDMARK_MODEL.exists():
            try:
                self.landmark_predictor = dlib.shape_predictor(str(Config.LANDMARK_MODEL))
            except Exception as e:
                print(f"Failed to load landmark model: {str(e)}")

    def detect_faces(self, frame):
        """
        检测视频帧中的人脸
        :param frame: BGR格式的视频帧
        :return: (face_locations, face_encodings) 人脸位置和编码
        """
        self.frame_count += 1

        # 跳帧处理
        if self.frame_count % Config.FRAME_SKIP != 0:
            return [], []

        # 转换颜色空间 BGR -> RGB
        rgb_frame = frame[:, :, ::-1]

        # 检测人脸位置
        face_locations = face_recognition.face_locations(
            rgb_frame,
            model=Config.FACE_DETECTION_MODEL,
            number_of_times_to_upsample=Config.UPSAMPLE
        )

        # 过滤太小的人脸
        face_locations = [
            loc for loc in face_locations
            if (loc[2] - loc[0]) >= Config.MIN_FACE_SIZE and
               (loc[1] - loc[3]) >= Config.MIN_FACE_SIZE
        ]

        # 提取人脸特征
        face_encodings = face_recognition.face_encodings(
            rgb_frame,
            face_locations,
            num_jitters=Config.NUM_JITTERS
        )

        return face_locations, face_encodings

    def detect_faces_with_landmarks(self, image):
        """
        检测人脸并返回面部特征点
        :param image: RGB图像
        :return: (face_locations, landmarks_list)
        """
        if self.landmark_predictor is None:
            return [], []

        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 检测人脸
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray, 1)
        face_locations = []
        landmarks_list = []

        for face in faces:
            # 转换dlib矩形为(top, right, bottom, left)
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            left = face.left()

            # 过滤太小的人脸
            if (bottom - top) < Config.MIN_FACE_SIZE or (right - left) < Config.MIN_FACE_SIZE:
                continue

            face_locations.append((top, right, bottom, left))

            # 获取特征点
            landmarks = self.landmark_predictor(gray, face)
            landmarks_list.append({
                "left_eye": (landmarks.part(36).x, landmarks.part(36).y),
                "right_eye": (landmarks.part(45).x, landmarks.part(45).y),
                "nose": (landmarks.part(30).x, landmarks.part(30).y),
                "mouth_left": (landmarks.part(48).x, landmarks.part(48).y),
                "mouth_right": (landmarks.part(54).x, landmarks.part(54).y),
                "jaw": [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)]
            })

        return face_locations, landmarks_list

    def draw_face_boxes(self, frame, face_locations, names=None):
        """
        在图像上绘制人脸框和姓名
        """
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # 绘制人脸框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # 如果有姓名则绘制
            if names and i < len(names):
                cv2.rectangle(
                    frame,
                    (left, bottom - 35),
                    (right, bottom),
                    (0, 255, 0),
                    cv2.FILLED
                )
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    frame,
                    names[i],
                    (left + 6, bottom - 6),
                    font,
                    0.8,
                    (0, 0, 0),
                    1
                )
        return frame

    def draw_landmarks(self, image, landmarks):
        """绘制面部特征点"""
        for landmark in landmarks:
            # 绘制眼睛
            cv2.circle(image, landmark["left_eye"], 2, (0, 255, 255), -1)
            cv2.circle(image, landmark["right_eye"], 2, (0, 255, 255), -1)

            # 绘制嘴巴
            cv2.line(image, landmark["mouth_left"], landmark["mouth_right"], (0, 255, 255), 1)

            # 绘制下巴
            for i in range(1, len(landmark["jaw"])):
                cv2.line(image, landmark["jaw"][i - 1], landmark["jaw"][i], (0, 255, 255), 1)

        return image