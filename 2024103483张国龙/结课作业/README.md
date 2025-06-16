# 图像去模糊深度学习项目

这是一个基于深度学习的图像去模糊项目，使用PyTorch框架实现。

## 项目结构

```
.
├── data_loader.py    # 数据加载和预处理模块
├── model.py         # 模型定义
├── train.py         # 训练脚本
├── evaluate.py      # 评估脚本
├── inference.py     # 推理脚本
├── requirements.txt # 项目依赖
└── README.md        # 项目说明文档
```

## 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- 其他依赖见 requirements.txt

## 安装

1. 克隆项目
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

将数据集按以下结构组织：
```
data/
├── blur/    # 模糊图像
└── sharp/   # 清晰图像
```

## 使用方法

### 训练模型

```bash
python train.py
```

### 评估模型

```bash
python evaluate.py
```

### 图像去模糊

```bash
python inference.py
```

## 模型架构

本项目使用了一个基于编码器-解码器结构的卷积神经网络，包含以下主要组件：
- 编码器：4个卷积块，逐步增加通道数
- 解码器：4个卷积块，逐步减少通道数
- 使用BatchNorm和ReLU激活函数
- 最终输出层使用Tanh激活函数

## 评估指标

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

## 注意事项

1. 训练前请确保有足够的GPU内存
2. 可以根据需要调整模型参数和训练参数
3. 建议使用高质量的数据集进行训练 