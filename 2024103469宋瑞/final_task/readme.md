# 基于U-Net的2018数据科学杯细胞核分割项目

## 1. 项目概述 (Project Overview)

本项目是针对 **[2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018)** 竞赛的解决方案，旨在实现显微镜图像中细胞核的**实例分割 (Instance Segmentation)**。我们采用深度学习方法，基于 **PyTorch** 框架实现了一个经典的 **U-Net** 模型，并构建了一套从数据处理、模型训练、性能监控到预测后处理和结果生成的完整工作流。

该项目代码结构清晰、模块化，并集成了多项工业界和学术界常用的技术，使其成为一个高质量的深度学习课程项目。

---

## 2. 项目特点 (Features)

- **模块化代码结构**: 代码被清晰地组织在不同的模块中 (`dataset.py`, `model.py`, `train.py` 等)，易于阅读和维护。
- **高级数据增强**: 使用业界领先的 **`albumentations`** 库进行复杂的数据增强，有效提升模型的泛化能力。
- **U-Net模型实现**: 从零开始实现了经典的U-Net架构，这是生物医学图像分割任务的基准模型。
- **高效训练策略**: 集成了**混合精度训练 (AMP)**、**学习率调度器 (ReduceLROnPlateau)** 和 **TensorBoard** 可视化监控，以实现高效、稳定的模型训练。
- **端到端推理流程**: 包含了完整的预测脚本，不仅能生成分割掩码，还包括**后处理**步骤（如孔洞填充、小对象移除）以优化结果。
- **生成竞赛提交文件**: 实现了从语义分割到实例分割的转换，并将结果编码为竞赛要求的**RLE (Run-Length Encoding)** 格式，生成 `submission.csv` 文件。

---

## 3. 项目结构 (Project Structure)

```
dsb2018_unet/
├── data/                 # 存放解压后的数据集
│   ├── stage1_train/
│   └── stage1_test/
├── checkpoints/          # 用于保存训练好的模型
├── logs/                 # TensorBoard 日志目录
├── config.py             # 配置文件，集中管理所有超参数和路径
├── dataset.py            # 数据加载和预处理模块 (使用Albumentations)
├── model.py              # U-Net模型定义
├── utils.py              # 工具函数（如指标计算、模型保存）
├── train.py              # 主训练脚本
├── inference_submit.py   # 预测、后处理并生成提交文件的脚本
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明文档
```

---

## 4. 技术方案 (Technical Approach)

### 4.1 数据处理

- **自定义`Dataset`**: `dataset.py`中的`BowlDataset`类负责加载图像，并将每个图像对应的多个二值掩码（每个文件一个细胞核）合并成一个单一的分割图。
- **数据增强**: 借助`albumentations`，在训练时对图像和掩码同步应用随机旋转、翻转等变换，有效扩充了数据集。

### 4.2 模型架构

- **U-Net**: `model.py`中实现了U-Net模型。其**编码器-解码器**结构和**跳跃连接 (Skip Connections)** 的设计，使其能够同时捕捉图像的深层语义信息和浅层细节特征，非常适合像素级的分割任务。

### 4.3 训练过程

- **损失函数**: 采用`BCEWithLogitsLoss`，它在数值上比`Sigmoid` + `BCELoss`更稳定。
- **优化器与调度器**: 使用`Adam`优化器和`ReduceLROnPlateau`学习率调度器，当模型性能在验证集上停滞时自动调整学习率。
- **监控**: 训练期间所有关键指标（Loss, Dice Score, IoU）都会被记录到`logs/`目录，可通过TensorBoard实时查看。

### 4.4 推理与后处理

- **实例分割**: 模型本身输出的是语义分割图。在`inference_submit.py`中，我们使用`skimage.morphology.label`对二值化的预测结果进行连通域分析，从而将不同的细胞核实例分离开。
- **后处理**: 为了提升分割质量，我们应用了**孔洞填充**和**小连通域移除**等形态学操作，以平滑掩码并去除噪声。
- **RLE编码**: 将每个分割出的细胞核实例编码为竞赛要求的RLE格式，生成最终的提交文件。

---

## 5. 如何运行 (How to Run)

### 步骤1: 环境配置

首先，克隆本仓库并安装所需依赖

### 步骤2: 准备数据

1. 从Kaggle竞赛页面下载数据集：[2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018/data)。
2. 解压 `stage1_train.zip` 和 `stage1_test.zip`。
3. 将解压后的 `stage1_train` 和 `stage1_test` 文件夹移动到 `data/` 目录下。

### 步骤3: 启动TensorBoard (可选)

为了实时监控训练过程，可以在项目根目录打开一个**新的终端**窗口，并运行：

```bash
tensorboard --logdir=logs
```

然后在浏览器中打开 `http://localhost:6006`。

### 步骤4: 开始训练

在主终端窗口中运行训练脚本。脚本将自动从训练集中划分出一部分作为验证集。

```bash
python train.py
```

训练过程中，性能最佳的模型（基于验证集IoU）将被保存在 `checkpoints/best_model.pth.tar`。

### 步骤5: 预测并生成提交文件

训练完成后，运行推理脚本。它将加载训练好的最佳模型，对测试集进行预测，并生成 `submission.csv` 文件。

```bash
python inference_submit.py
```
