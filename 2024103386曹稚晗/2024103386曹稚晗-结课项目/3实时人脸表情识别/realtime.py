import cv2
import numpy as np
from keras.models import load_model


class RealTimeEmotionDetection:
    def __init__(self):
        # 加载人脸检测器和表情识别模型
        self.face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.emotion_model = load_model('models/emotion_model.hdf5')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        # 颜色设置
        self.colors = {
            'Angry': (0, 0, 255),
            'Disgust': (0, 102, 102),
            'Fear': (102, 0, 102),
            'Happy': (0, 255, 255),
            'Sad': (255, 0, 0),
            'Surprise': (0, 255, 0),
            'Neutral': (255, 255, 255)
        }

    def preprocess_face(self, face_img):
        # 转换为灰度图并调整大小
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

        # 归一化并添加维度
        normalized = resized.astype('float32') / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))

        return reshaped

    def detect_emotion(self, frame):
        # 转换为灰度图用于人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_img = frame[y:y + h, x:x + w]

            # 预处理人脸图像
            processed_face = self.preprocess_face(face_img)

            # 预测表情
            predictions = self.emotion_model.predict(processed_face)
            emotion_idx = np.argmax(predictions)
            emotion = self.emotion_labels[emotion_idx]
            confidence = np.max(predictions)

            # 绘制边界框和表情标签
            color = self.colors[emotion]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = f"{emotion} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame

    def run(self):
        # 打开摄像头
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("无法打开摄像头")
            return

        # 设置窗口
        cv2.namedWindow('Real-time Emotion Detection', cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测表情
            processed_frame = self.detect_emotion(frame)

            # 显示结果
            cv2.imshow('Real-time Emotion Detection', processed_frame)

            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = RealTimeEmotionDetection()
    detector.run()