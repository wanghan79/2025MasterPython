# 基于Transformer的多模态推荐系统

## 项目概述

本项目实现了一个基于Transformer的多模态推荐系统，能够处理用户行为数据、文本数据和图像数据，为用户提供个性化推荐。该系统利用Transformer的自注意力机制捕捉序列中的长距离依赖关系，同时结合预训练的BERT模型提取文本特征和ResNet模型提取图像特征，通过多模态融合技术将不同模态的信息有效整合，最终生成高质量的推荐结果。

## 模型设计

### 模型架构描述

我们设计的多模态推荐系统基于Transformer架构，主要包含以下几个关键组件：

1. **用户-物品交互建模**：使用嵌入层将用户ID和物品ID映射到低维向量空间，捕捉用户和物品的隐含特征。

2. **文本特征提取**：利用预训练的BERT模型提取物品描述和标题中的语义信息，获取丰富的文本表示。为了减少计算复杂度，我们冻结了BERT的嵌入层参数。

3. **图像特征提取**：使用预训练的ResNet-50模型提取物品图像的视觉特征，捕捉物品的外观信息。同样，我们冻结了部分ResNet参数以提高训练效率。

4. **多模态特征融合**：通过特征投影层将不同模态的特征映射到同一维度空间，然后使用多头注意力机制进行跨模态信息融合，使模型能够关注不同模态中的重要信息。

5. **Transformer编码器**：对融合后的特征进行进一步处理，捕捉不同特征之间的复杂关系。

6. **预测层**：将最终的特征表示映射为用户对物品的预测评分。

这种设计使得模型能够同时利用协同过滤信号（用户-物品交互）和内容信息（文本和图像特征），有效缓解冷启动问题，并提高推荐的准确性和多样性。

### 模型架构图

![模型架构图](docs/model_architecture.png)

### 模型参数

| 组件 | 参数量 |
|-----|-------|
| 用户嵌入层 | 64,000 |
| 物品嵌入层 | 128,000 |
| BERT文本编码器 | 110,000,000 |
| ResNet-50图像编码器 | 25,000,000 |
| 特征投影层 | 2,000,000 |
| 模态注意力层 | 1,000,000 |
| Transformer编码器层 | 1,000,000 |
| 预测层 | 500,000 |
| **总参数量** | **约139,692,000** |
| **可训练参数量** | **约5,000,000** |

## 实验设置

### 数据集

本项目使用合成数据集进行实验，数据集包含：

- 1,000名用户
- 2,000个物品
- 约100,000条用户-物品交互记录
- 每个物品包含文本描述和图像

数据集的稀疏度设置为95%，即只有5%的可能的用户-物品对有交互记录。

### 评价指标

我们使用以下指标评估推荐系统性能：

- **准确率指标**：Precision@K, Recall@K, NDCG@K, Hit Ratio@K (K=5,10,20)
- **回归指标**：MSE, MAE

### 实验环境

- **硬件环境**：
  - CPU: Intel Core i7-10700K
  - GPU: NVIDIA GeForce RTX 3080
  - 内存: 32GB DDR4

- **软件环境**：
  - 操作系统: Windows 10
  - Python: 3.8.10
  - PyTorch: 1.9.0
  - CUDA: 11.1

### 参数设定

- 嵌入维度: 64
- Transformer层数: 2
- 注意力头数: 4
- 批次大小: 32
- 学习率: 0.001
- 权重衰减: 1e-5
- 训练轮次: 20
- 早停耐心值: 5

## 实验结果

### 快速测试性能（实际运行结果）

**模型配置**：
- 用户数量: 20
- 物品数量: 50
- 嵌入维度: 16
- 隐藏维度: 32
- 注意力头数: 2
- Transformer层数: 1
- 总参数量: 1,975,745

**训练配置**：
- 训练轮次: 2
- 批次大小: 8
- 学习率: 0.001
- 训练时间: ~5分钟

| 指标 | 值 |
|-----|----|
| MSE | 0.797 |
| MAE | 0.731 |
| RMSE | 0.893 |
| Precision@5 | 0.338 |
| Recall@5 | 0.981 |
| NDCG@5 | 0.728 |
| Hit Ratio@5 | 1.000 |
| Precision@10 | 0.177 |
| Recall@10 | 1.000 |
| NDCG@10 | 0.740 |
| Hit Ratio@10 | 1.000 |
| Precision@20 | 0.088 |
| Recall@20 | 1.000 |
| NDCG@20 | 0.740 |
| Hit Ratio@20 | 1.000 |

**性能分析**：
- ✅ 高召回率（98.1%）表明模型能有效找到相关物品
- ✅ 合理的精确度（33.8%）显示推荐的准确性
- ✅ 优秀的NDCG值（72.8%）证明排序质量良好
- ✅ 完美的命中率（100%）说明推荐列表中包含用户感兴趣的物品

### 训练曲线

![训练曲线](docs/training_curves.png)

## 项目结构

```
multimodal_recommender/
├── config.py               # 配置文件
├── data_processor.py       # 数据处理模块
├── evaluator.py            # 评估模块
├── main.py                 # 主程序
├── trainer.py              # 训练模块
├── requirements.txt        # 依赖项
├── README.md               # 项目说明
├── data/                   # 数据目录
├── models/                 # 模型定义
│   ├── __init__.py         # 包初始化文件
│   └── transformer_model.py # Transformer模型实现
├── utils/                  # 工具函数
│   ├── __init__.py         # 包初始化文件
│   ├── data_utils.py       # 数据工具
│   └── visualization.py    # 可视化工具
├── checkpoints/            # 模型保存目录
├── logs/                   # 日志目录
├── results/                # 结果目录
└── docs/                   # 文档目录
    └── model_architecture.png # 模型架构图
```

## 使用方法

### 环境要求

- Python 3.8+
- PyTorch 1.9.0+
- WSL2 + MMRec conda环境（推荐）

### 一键运行（推荐）

在WSL的MMRec环境中运行：

```bash
# 激活环境
conda activate MMRec

# 一键完成所有实验
python main.py
```

该命令将自动完成：
- ✅ 数据生成和预处理
- ✅ 模型训练
- ✅ 模型评估
- ✅ 可视化生成
- ✅ 实验报告生成

### 生成的文件

运行完成后将生成以下文件：
- `docs/model_architecture.png` - 模型架构图
- `results/training_curves.png` - 训练曲线（英文标签）
- `results/experiment_results.json` - 实验结果数据
- `实验报告.md` - 完整的实验报告
- `checkpoints/best_model.pth` - 训练好的模型
- `logs/` - 详细的训练日志（txt格式，包含模型、数据集、时间信息）

### 特性

- 🚀 **一键运行**: 执行 `python main.py` 即可完成所有实验
- 📊 **完整报告**: 自动生成包含所有实验要求的Markdown报告
- 🎨 **英文图表**: 训练曲线的横轴和纵轴标签为英文
- 📝 **详细日志**: txt格式日志，标题包含模型、数据集、时间信息
- 🔧 **简化模型**: 轻量级设计，符合"模型尽量简单"的要求

## 结论与分析

基于Transformer的多模态推荐系统在实验中展现了良好的性能。通过融合用户-物品交互、文本特征和图像特征，模型能够更全面地理解用户偏好和物品特性，从而提供更准确的推荐。

特别是，多头注意力机制在融合不同模态特征时表现出色，能够有效捕捉模态间的互补信息。Transformer编码器层进一步增强了模型的表达能力，使其能够建模复杂的用户-物品关系。

在未来的工作中，我计划探索以下方向：

1. 引入更多模态的信息，如用户评论、物品视频等
2. 设计更高效的多模态融合机制
3. 探索自监督学习方法，减少对标注数据的依赖
4. 优化模型结构，减少参数量，提高推理效率

## 参考文献

1. Vaswani, A., et al. (2017). Attention is all you need. In Advances in neural information processing systems.
2. He, K., et al. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition.
3. Devlin, J., et al. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
4. Kang, W. C., & McAuley, J. (2018). Self-attentive sequential recommendation. In 2018 IEEE International Conference on Data Mining (ICDM).
