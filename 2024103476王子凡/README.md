# Python数据挖掘


本项目包含三个独立的作业，分别实现了Python数据结构的性能测试、随机数据生成和统计分析功能。

## 项目结构
```
.
├── assignment1/          # 作业一：数据结构性能测试
│   ├── performance_test.py
│   └── README.md
├── assignment2/          # 作业二：随机数据生成器
│   ├── data_sampler.py
│   └── README.md
└── assignment3/          # 作业三：统计装饰器
    ├── stats_decorator.py
    └── README.md
```

## 作业说明

### 作业一：数据结构性能测试
- 对比list和tuple在修改操作时的性能差异
- 创建10000×10000的数据矩阵
- 进行10000次随机位置修改
- 详细说明见 [作业一README](assignment1/README.md)

### 作业二：随机数据生成器
- 实现灵活的数据生成器
- 支持多种数据类型的随机生成
- 支持任意嵌套的数据结构
- 详细说明见 [作业二README](assignment2/README.md)

### 作业三：统计装饰器
- 实现带参数的装饰器
- 支持多种统计指标计算
- 与作业二的数据生成器集成
- 详细说明见 [作业三README](assignment3/README.md)

## 运行环境要求
- Python 3.6+
- numpy

## 安装依赖
```bash
pip install numpy
```

## 使用说明
每个作业都是独立的，可以单独运行。请参考各个作业目录下的README文件了解具体使用方法。

## 注意事项
- 作业一需要较大的内存空间
- 作业二和作业三可以组合使用
- 所有代码都包含详细的注释和文档

## 提交说明
- 截止日期：2024年6月30日晚24点
- 提交方式：GitHub仓库
- 提交内容：完整的代码和文档 