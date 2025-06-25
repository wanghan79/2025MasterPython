import torch  # 导入 PyTorch 库
import torch.optim as optim  # 导入优化器模块
import torch.nn as nn  # 导入神经网络模块
import numpy as np  # 导入 NumPy 库，用于数值计算
import matplotlib.pyplot as plt  # 导入 Matplotlib，用于绘图
import os  # 导入 OS 库，用于文件操作
from sklearn.model_selection import train_test_split  # 导入数据集划分工具
from torch.utils.data import DataLoader, TensorDataset  # 导入数据加载器和数据集工具
from models.FaceCNN import FaceCNN  # 引入自定义的 CNN 模型（假设已定义）

# 加载数据（假设已经准备好数据）
X_train = np.load('X_train.npy')  # 加载训练数据
X_test = np.load('X_test.npy')  # 加载测试数据
y_train = np.load('y_train.npy')  # 加载训练标签
y_test = np.load('y_test.npy')  # 加载测试标签

# 数据归一化（像素值归一化到 [0, 1] 范围）
X_train = X_train / 255.0  # 训练数据归一化
X_test = X_test / 255.0  # 测试数据归一化

# 转换为 torch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # 转换为 Tensor，并添加频道维度
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # 转换为 Tensor，并添加频道维度
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # 训练标签转换为 Tensor
y_test_tensor = torch.tensor(y_test, dtype=torch.long)  # 测试标签转换为 Tensor

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # 创建训练数据集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 创建测试数据集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 创建训练数据加载器
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 创建测试数据加载器

# 定义模型、损失函数和优化器
model = FaceCNN(num_classes=31).cuda()  # 初始化自定义 CNN 模型，假设共有 31 类
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数（适用于分类任务）
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 使用 Adam 优化器，初始学习率为 0.0005

# 创建学习率调度器（每30个epoch降低学习率，衰减因子为0.1）
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 每30个 epoch 更新一次学习率，衰减系数为 0.1

# 创建保存权重的文件夹
if not os.path.exists('weights'):  # 如果文件夹不存在
    os.makedirs('weights')  # 创建一个名为 'weights' 的文件夹

# 训练循环
epochs = 200  # 训练 200 个 epoch
train_losses = []  # 存储每个 epoch 的训练损失
test_losses = []  # 存储每个 epoch 的测试损失
train_accuracies = []  # 存储每个 epoch 的训练准确率
test_accuracies = []  # 存储每个 epoch 的测试准确率

for epoch in range(epochs):  # 循环 200 次，每次训练一个 epoch
    model.train()  # 设置模型为训练模式
    running_train_loss = 0.0  # 初始化训练损失
    correct_train = 0  # 初始化正确预测的数量
    total_train = 0  # 初始化总的样本数

    for inputs, labels in train_loader:  # 遍历训练数据集
        inputs, labels = inputs.cuda(), labels.cuda()  # 将数据移至 GPU（假设使用 CUDA）

        optimizer.zero_grad()  # 清除梯度信息

        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        running_train_loss += loss.item()  # 累加训练损失
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total_train += labels.size(0)  # 统计样本总数
        correct_train += (predicted == labels).sum().item()  # 统计正确的预测数

    train_losses.append(running_train_loss / len(train_loader))  # 记录训练损失
    train_accuracies.append(100 * correct_train / total_train)  # 记录训练准确率

    # 测试
    model.eval()  # 设置模型为评估模式
    running_test_loss = 0.0  # 初始化测试损失
    correct_test = 0  # 初始化测试正确预测的数量
    total_test = 0  # 初始化测试总样本数

    with torch.no_grad():  # 在测试时不需要计算梯度
        for inputs, labels in test_loader:  # 遍历测试数据集
            inputs, labels = inputs.cuda(), labels.cuda()  # 将数据移至 GPU
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            running_test_loss += loss.item()  # 累加测试损失
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total_test += labels.size(0)  # 统计样本总数
            correct_test += (predicted == labels).sum().item()  # 统计正确的预测数

    test_losses.append(running_test_loss / len(test_loader))  # 记录测试损失
    test_accuracies.append(100 * correct_test / total_test)  # 记录测试准确率

    # 打印当前 epoch 的损失、准确率和学习率
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")
    print(f"Train Accuracy: {train_accuracies[-1]:.2f}%, Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")  # 打印当前学习率

    # 保存每一层的权重
    for name, param in model.named_parameters():  # 遍历模型的每个参数
        if 'weight' in name:  # 只保存包含 'weight' 的层（即卷积层和全连接层）
            weight_file = os.path.join('weights', f'{name}_epoch_{epoch + 1}.pt')  # 设置文件名
            torch.save(param.data.cpu(), weight_file)  # 保存权重到文件

    # 保存模型权重
    torch.save(model.state_dict(), f'face_cnn_model_epoch_{epoch + 1}.pth')  # 保存整个模型的权重

    # 更新学习率
    scheduler.step()  # 根据学习率调度器更新学习率

# 绘制训练和测试误差曲线
plt.plot(range(epochs), train_losses, label="Train Loss")  # 绘制训练损失
plt.plot(range(epochs), test_losses, label="Test Loss")  # 绘制测试损失
plt.xlabel("Epoch")  # 设置 x 轴标签
plt.ylabel("Loss")  # 设置 y 轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图像

# 绘制训练和测试准确率曲线
plt.plot(range(epochs), train_accuracies, label="Train Accuracy")  # 绘制训练准确率
plt.plot(range(epochs), test_accuracies, label="Test Accuracy")  # 绘制测试准确率
plt.xlabel("Epoch")  # 设置 x 轴标签
plt.ylabel("Accuracy (%)")  # 设置 y 轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图像

# 可视化第一个卷积层的滤波器
first_conv_layer = model.conv1  # 假设第一个卷积层为 'conv1'
filters = first_conv_layer.weight.data.cpu().numpy()  # 获取该层的滤波器权重

# 绘制滤波器
num_filters = filters.shape[0]  # 获取滤波器数量
fig, axes = plt.subplots(1, num_filters, figsize=(num_filters * 2, 2))  # 创建子图
for i, ax in enumerate(axes):  # 遍历每个滤波器
    ax.imshow(filters[i, 0, :, :], cmap='gray')  # 只绘制每个滤波器的第一通道
    ax.axis('off')  # 隐藏坐标轴
plt.show()  # 显示图像