# UNet Cityscapes 语义分割项目

## 简介
本项目基于 PyTorch 实现 UNet 语义分割模型，支持 Cityscapes 数据集，支持 GPU 训练与 TensorBoard 可视化。

## 依赖安装
```bash
pip install -r requirements.txt
```

## 数据集准备
1. 下载 [Cityscapes 数据集](https://www.cityscapes-dataset.com/downloads/)。
2. 解压后，将 `leftImg8bit` 和 `gtFine` 文件夹放入 `./data/Cityscapes/` 目录下。

目录结构示例：
```
./data/Cityscapes/
    leftImg8bit/
        train/
        val/
    gtFine/
        train/
        val/
```

## 训练
```bash
python train.py --config config.py
```

## 推理与可视化
```bash
python predict.py --config config.py --image_path path/to/image.png
```

## TensorBoard 可视化
```bash
tensorboard --logdir=logs
```

## 主要文件说明
- config.py：配置文件
- dataset.py：数据集加载与增强
- model.py：UNet模型
- loss.py：损失函数
- metrics.py：评价指标
- train.py：训练与验证主循环
- utils.py：工具函数
- predict.py：推理与可视化 