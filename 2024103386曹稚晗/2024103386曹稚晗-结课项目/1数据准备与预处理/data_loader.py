import numpy as np
import cv2
import os
from keras.utils import to_categorical


class DataLoader:
    def __init__(self, dataset_path, target_size=(48, 48)):
        self.dataset_path = dataset_path
        self.target_size = target_size
        self.emotions = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'sad': 4, 'surprise': 5, 'neutral': 6
        }

    def load_data(self):
        faces = []
        labels = []

        for emotion_name, emotion_idx in self.emotions.items():
            emotion_path = os.path.join(self.dataset_path, emotion_name)

            for image_name in os.listdir(emotion_path):
                image_path = os.path.join(emotion_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is not None:
                    # 预处理图像
                    image = cv2.resize(image, self.target_size)
                    image = image.astype('float32') / 255.0
                    image = np.expand_dims(image, axis=-1)  # 添加通道维度

                    faces.append(image)
                    labels.append(emotion_idx)

        # 转换为numpy数组并one-hot编码
        faces = np.array(faces)
        labels = to_categorical(np.array(labels))

        return faces, labels