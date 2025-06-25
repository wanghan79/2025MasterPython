# 汉字识别YOLOv11项目

## 运行环境配置

1. Python环境:
```bash
conda create -n yolov11 python=3.11
conda activate yolov11
```

2. 安装PyTorch (CUDA 12.6版本):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

3. 安装项目依赖:
```bash
pip install -r requirements.txt
```

4. 验证CUDA可用性:
```python
import torch
print(torch.cuda.is_available())  # 应输出True
print(torch.cuda.device_count())  # 应输出GPU数量
```

## 项目概述
本项目基于YOLOv11实现汉字识别功能，包含完整的数据集准备、模型训练、测试评估和预测推理流程。

## 文件结构说明
```
.
├── get_dataset.py       # 数据集准备脚本
├── load_model.py        # 模型加载工具，包括直接加载yolo模型或者选择加载底层的torch.nn.Module模型
├── train.py            # 模型训练脚本
├── test.py             # 模型测试脚本  
├── predict.py          # 预测推理脚本
├── yolo11n.pt          # 预训练模型
|── best_model.pt        # 训练得到的最佳模型，对应为train9的得到的最好的模型
├── ultralytics
    |—— cfg/
        └── default.yaml # 训练配置文件               
│   ├── detect/# 训练结果和预测输出
│   │   ├── train*/     # 训练、测试集测试日志和权重
│   │   └── predict/    # 预测结果
└──
```

## 数据集准备
- 相应的原始数据在https://huggingface.co/AISkywalker对应的Datasets下载
1. 准备数据目录结构：
```
Data/
└── data/
    ├── 00001/  # 以Unicode编码命名的文件夹
    │   ├── 1.jpg
    │   └── ...
    ├── 00002/
    └── ...
```

2. 准备字符映射文件`char_dict.json`，格式示例：
```json
{
    "1": "一",
    "2": "二",
    ...
}
```

3. 运行数据准备脚本：
```bash
python get_dataset.py
```
脚本将自动：
- 扫描图像文件并匹配类别
- 划分训练集/验证集/测试集
- 生成YOLO格式数据集和dataset.yaml，文件保存在项目同一级的根目录yolo_hanzi_dataset/
- 特别注意您的原始数据集路径和字符映射文件路径，以及yolo格式的数据路径，因为这个路径在训练、测试和预测过程中都会用到。

## 模型训练
1. 修改训练配置：
编辑`ultralytics/cfg/default.yaml`调整超参数

2. 开始训练：
```bash
conda activate yolov11
python train.py
```
训练过程将：
- 加载预训练模型yolo11n.pt
- 使用dataset.yaml配置数据集
- 保存训练日志和权重到runs/detect/train*
- 注意您的训练配置文件以及数据配置文件路径

## 测试评估
```bash
conda activate yolov11
python test.py
```
测试将：
- 加载最佳模型(runs/detect/train*/weights/best.pt)
- 在测试集上评估模型性能
- 输出mAP指标和可视化结果
- 注意您的测试配置文件路径以及测试模型路径

## 预测推理
```bash
conda activate yolov11
python predict.py
```
预测功能：
- 加载训练好的模型
- 对单张图片进行预测
- 显示和保存预测结果
- 打印检测到的汉字类别和置信度
- 注意您的预测图像路径以及预测模型路径

## 常见问题
1. 数据集准备失败：
- 检查数据目录结构是否正确
- 确认char_dict.json编码和格式正确
- 确保图像文件具有读取权限
- 检查数据集路径配置正确

2. 训练过程中断：
- 检查CUDA内存是否不足，可减小batch size以及图片size
- 确保数据集路径配置正确

3. 训练测试效果不佳：
- 模型过拟合：尝试使用更小的batch size和epochs，增大早停止的周期数。
- 模型欠拟合：尝试使用更复杂的模型架构，汉字识别项目数据增强可能过度，导致不必要的增强。增加图片size大小可能能提高性能。

4. 预测结果不准确：
- 检查训练数据是否充足
- 尝试调整置信度阈值(修改predict.py中的conf参数)
- 考虑使用更精确的标注数据

## 性能指标
典型训练结果：
- mAP@0.5: 0.90-0.95
- mAP@0.5:0.95: 0.90-0.95

![训练效果图](ultralytics/runs/detect/train9/results.png)

典型测试结果：
![测试效果图](ultralytics/runs/detect/val9/val_batch2_pred.jpg)
![测试效果图](ultralytics/runs/detect/val9/confusion_matrix.png)

典型预测结果：
![预测效果图](ultralytics/runs/detect/predict2/image0.jpg)
![预测效果图](ultralytics/runs/detect/predict2/image0.jpg)



## 后续改进
1. 该模型的预测精度高达0.97，后期可以尝试使用更小的模型————在保证精度可控的范围内，缩短训练以及推理时间。
2. 可以扩展到边缘设备，增加模型部署的效率。
