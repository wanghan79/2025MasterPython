import cv2
import time
from typing import Optional, Union


class VideoCapture:
    """视频捕获工具类"""

    def __init__(self, source: Union[int, str] = 0):
        """
        初始化视频源
        :param source: 视频源(摄像头索引或文件路径)
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        # 获取视频属性
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        return frame

    def get_frame_size(self) -> Tuple[int, int]:
        """获取帧尺寸"""
        return (self.width, self.height)

    def set_resolution(self, width: int, height: int):
        """设置分辨率"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height

    def release(self):
        """释放资源"""
        self.cap.release()

    def __del__(self):
        self.release()