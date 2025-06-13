# README.md

项目名称：基于BERT的中文情感分析系统

项目简介：
使用预训练的BERT模型对中文评论进行情感分类，实现一个端到端的NLP训练、评估和推理流程。

主要功能：
1. 数据加载与预处理（tokenization, padding）
2. 模型封装（加载预训练模型、添加分类层）
3. 训练脚本（支持 checkpoint、early stopping）
4. 评估脚本（计算准确率、F1）
5. 推理脚本（命令行接口）

代码文件：
- data_loader.py: 数据集读入与处理
- model.py: 模型定义
- train.py: 训练流程
- evaluate.py: 模型评估
- predict.py: 单句预测
- utils.py: 工具函数（日志、保存加载）
