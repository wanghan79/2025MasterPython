from ultralytics import YOLO
import cv2
from load_model import load_model 
#对单张图片进行预测
def predict(model,image_path):
    image = cv2.imread(image_path)#加载图片
    model = load_model(model)#加载yolo模型
    results = model.predict(source=image, 
                        save=True,  # 保存预测结果
                        show=True,   # 显示预测结果
                        conf=0.5)    # 置信度阈值
    # 4. 打印预测结果
    for result in results:
        print("检测到的对象:")
        for box in result.boxes:
            print(f"- 类别: {result.names[box.cls[0].item()]}, 置信度: {box.conf[0].item():.2f}")
    
if __name__ == '__main__':
    predict(r'runs\detect\train9\weights\best.pt',r'..\yolo_hanzi_dataset\images\test\00800_243401.png')
