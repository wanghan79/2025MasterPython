import torch  # 导入 PyTorch 库，用于深度学习操作
import numpy as np  # 导入 NumPy 库，用于数组处理
import os  # 导入 os 库，用于操作系统文件和路径
import matplotlib.pyplot as plt  # 导入 Matplotlib 库，用于绘制图表
import seaborn as sns  # 导入 Seaborn 库，用于更美观的图表绘制
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # 导入 scikit-learn 库，用于计算模型性能指标
from models.FaceCNN import FaceCNN  # 从自定义模型文件导入 FaceCNN 模型（根据实际路径修改）

# 定义加载模型的函数
def load_model(model_path='pth(lr=0.0005)/face_cnn_model_epoch_200.pth', num_classes=31):
    model = FaceCNN(num_classes=num_classes)  # 创建一个 FaceCNN 模型实例，指定类别数
    model.load_state_dict(torch.load(model_path, map_location='cpu'))  # 加载保存的模型权重到模型中
    model.eval()  # 设置模型为评估模式（例如关闭 dropout）
    return model  # 返回加载好的模型

# 定义加载数据的函数
def load_data():
    X_test = np.load('X_test.npy') / 255.0  # 加载测试数据并将像素值归一化到 [0, 1] 范围
    y_test = np.load('y_test.npy')  # 加载对应的标签数据
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # 将输入数据转换为 PyTorch 张量，并添加一个通道维度，形状变为 (N, 1, 64, 64)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)  # 将标签数据转换为 PyTorch 长整型张量
    return X_test_tensor, y_test_tensor  # 返回转换后的测试数据和标签

# 定义评估模型性能的函数
def evaluate(model, X_test, y_test):
    with torch.no_grad():  # 在评估阶段禁用梯度计算（节省内存和计算资源）
        outputs = model(X_test)  # 将测试数据传入模型，得到模型输出
        _, predicted = torch.max(outputs, 1)  # 获取每个样本的最大值索引作为预测类别
        acc = accuracy_score(y_test.numpy(), predicted.numpy())  # 计算准确率
        report = classification_report(y_test.numpy(), predicted.numpy())  # 生成分类报告，包含精度、召回率等
        cm = confusion_matrix(y_test.numpy(), predicted.numpy())  # 生成混淆矩阵
        top_k_acc = top_k_accuracy(outputs, y_test, k=5)  # 计算 Top-5 准确率
    return acc, report, cm, top_k_acc  # 返回准确率、分类报告、混淆矩阵和 Top-5 准确率

# 定义计算 Top-k 准确率的函数
def top_k_accuracy(outputs, targets, k=5):
    """ 计算Top-k准确率 """
    _, top_k_pred = torch.topk(outputs, k, dim=1, largest=True, sorted=False)  # 获取输出的前 k 个预测值
    correct = top_k_pred.eq(targets.view(-1, 1).expand_as(top_k_pred))  # 检查预测值是否正确
    return correct[:, :k].sum().item() / targets.size(0)  # 计算正确预测的比例并返回

# 定义保存混淆矩阵为图像的函数
def save_confusion_matrix(cm, filename='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))  # 设置图像大小
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')  # 使用 Seaborn 绘制混淆矩阵的热图，数字格式为整数
    plt.title("Confusion Matrix")  # 设置图像标题
    plt.xlabel("Predicted")  # 设置 x 轴标签
    plt.ylabel("True")  # 设置 y 轴标签
    plt.tight_layout()  # 自动调整布局
    plt.savefig(filename)  # 将图像保存为文件
    print(f"混淆矩阵保存为 {filename}")  # 输出保存成功的提示
    plt.close()  # 关闭图像窗口

# ✅ pytest 兼容测试函数
def test_model_performance():
    model_path = 'pth(lr=0.0005)/face_cnn_model_epoch_200.pth'  # 模型权重文件的路径
    assert os.path.exists(model_path), f"模型文件 {model_path} 不存在"  # 确认模型文件存在，否则抛出异常

    model = load_model(model_path)  # 加载模型
    X_test, y_test = load_data()  # 加载测试数据和标签
    acc, report, cm, top_k_acc = evaluate(model, X_test, y_test)  # 评估模型性能

    print(f"\n✅ 测试准确率: {acc * 100:.2f}%")  # 打印测试准确率
    print(f"✅ Top-5 准确率: {top_k_acc * 100:.2f}%")  # 打印 Top-5 准确率
    print("\n✅ 分类报告:\n", report)  # 打印分类报告

    save_confusion_matrix(cm)  # 保存混淆矩阵图像

    # 可以调整期望准确率阈值
    assert acc > 0.80, "模型准确率过低"  # 如果准确率低于 80%，则抛出异常
