import os  # 导入os模块，用于文件和目录操作
import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于数组操作
from sklearn.model_selection import train_test_split  # 从sklearn导入train_test_split，用于划分数据集

# 设置图像文件夹路径
image_folder = 'D:/Soft/PyCharm/WorkSpace/Pytorch2/FaceRecognize/data/Process'  # 定义图像数据存放文件夹路径

# 初始化图像数据和标签列表
X = []  # 存放图像数据的列表
y = []  # 存放标签的列表,每个人一个标签

# 遍历每个参与者的文件夹
for label, person_folder in enumerate(os.listdir(image_folder)):  # 遍历每个参与者的文件夹，label是标签
    person_folder_path = os.path.join(image_folder, person_folder)  # 获取每个参与者文件夹的完整路径

    # 确保是文件夹
    if os.path.isdir(person_folder_path):  # 判断是否为文件夹
        print(f"Processing folder: {person_folder} with label {label}")  # 打印当前正在处理的文件夹和对应的标签
        for image_name in os.listdir(person_folder_path):  # 遍历每个参与者文件夹中的图像文件
            image_path = os.path.join(person_folder_path, image_name)  # 获取图像的完整路径

            # 读取图像
            img = cv2.imread(image_path)  # 使用OpenCV读取图像

            # 检查图像是否加载成功
            if img is None:  # 如果图像加载失败
                print(f"Failed to load image {image_path}")  # 打印加载失败的图像路径
                continue  # 跳过加载失败的图像，继续处理下一个图像

            # 转换为灰度图像
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像从BGR格式转换为灰度图

            # 调整图像大小为 64x64（可以根据需要调整）
            img = cv2.resize(img, (64, 64))  # 将图像调整为64x64的大小

            # 将图像数据和标签添加到列表中
            X.append(img)  # 将处理后的图像添加到图像数据列表中
            y.append(label)  # 将当前参与者的标签添加到标签列表中（参与者的文件夹名作为标签）

# 转换为 numpy 数组
X = np.array(X)  # 将图像数据转换为NumPy数组
y = np.array(y)  # 将标签转换为NumPy数组

# 打印数据集信息
print(f"Loaded {len(X)} images with {len(set(y))} unique labels.")  # 打印加载的图像数量和唯一标签的数量
print(f"Labels: {y[:10]}")  # 打印前10个标签
print(f"X shape: {X.shape}, y shape: {y.shape}")  # 打印X和y的形状，检查数据集的维度

# 划分数据集（80% 训练集，20% 测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 使用train_test_split将数据集划分为训练集和测试集


# 保存处理后的数据为 .npy 文件
np.save(os.path.join('X_train.npy'), X_train)  # 将训练集图像数据保存为X_train.npy
np.save(os.path.join('X_test.npy'), X_test)  # 将测试集图像数据保存为X_test.npy
np.save(os.path.join('y_train.npy'), y_train)  # 将训练集标签保存为y_train.npy
np.save(os.path.join('y_test.npy'), y_test)  # 将测试集标签保存为y_test.npy

print("Data has been saved to .npy files in the output folder.")  # 打印数据保存成功的提示信息
