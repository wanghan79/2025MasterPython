# 目标检测项目

## 项目概述

本项目实现了一个完整的目标检测系统，能够识别图像或视频中的多个目标类别。系统基于PyTorch框架，使用预训练的Faster R-CNN模型作为检测器，并提供了可视化和评估功能。

## 主要功能

1. **目标检测**：使用预训练的Faster R-CNN模型检测图像或视频中的目标
2. **多类别支持**：默认支持COCO数据集的80个类别，也可以自定义类别
3. **可视化**：将检测结果可视化，显示边界框和类别标签
4. **评估**：计算检测结果的性能指标，如AP和mAP
5. **灵活配置**：通过配置文件可以调整各种参数

## 技术实现

本项目采用模块化设计，主要包含以下组件：

1. **主程序**：协调各个模块的工作流程
2. **检测器**：实现目标检测功能，基于PyTorch的Faster R-CNN
3. **可视化器**：将检测结果可视化
4. **数据集处理**：处理图像和视频输入
5. **评估器**：计算检测性能指标
6. **工具函数**：提供常用功能，如图像读取、IoU计算等

## 使用方法

### 安装依赖
pip install torch torchvision opencv-python numpy matplotlib pyyaml
### 运行检测
python main.py --input path/to/your/image/or/video --input_type image/video --visualize --evaluate
### 配置参数

可以通过修改`config.yaml`文件来调整各种参数，如模型类型、置信度阈值、可视化设置等。

## 项目结构
object-detection/
├── main.py                # 主程序入口
├── detector.py            # 目标检测器
├── visualizer.py          # 可视化模块
├── dataset.py             # 数据集处理模块
├── evaluator.py           # 评估模块
├── utils.py               # 工具函数
├── config.yaml            # 配置文件
├── readme.md              # 项目文档
└── logs/                  # 日志文件目录
└── results/               # 结果输出目录
## 性能评估

系统提供了完整的评估功能，可以计算各个类别的AP（Average Precision）和mAP（Mean Average Precision），并生成可视化图表。

## 扩展功能

1. 支持自定义模型和权重
2. 可以添加更多的可视化方式
3. 支持批量处理多个图像或视频
4. 可以扩展支持其他目标检测模型    