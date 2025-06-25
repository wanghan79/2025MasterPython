# Python数据统计装饰器

## 项目说明
本项目实现了一个带参数的装饰器，用于对随机生成的数据样本进行统计分析。装饰器可以计算多种统计指标，包括求和、均值、方差和均方根误差。

## 功能特点
- 支持多种统计指标：
  - SUM（求和）
  - AVG（均值）
  - VAR（方差）
  - RMSE（均方根误差）
- 可自定义需要计算的统计指标组合
- 支持任意嵌套数据结构的数值统计
- 与作业二的数据生成器无缝集成

## 运行环境要求
- Python 3.6+
- numpy

## 安装依赖
```bash
pip install numpy
```

## 使用方法
1. 定义数据结构：
```python
test_structure = {
    "id": int,
    "scores": [float, float, float],
    "nested": {
        "value": int,
        "array": [float, float]
    }
}
```

2. 使用装饰器：
```python
from stats_decorator import StatsDecorator

@StatsDecorator(['SUM', 'AVG', 'VAR', 'RMSE'])
def generate_test_data(structure, num_samples):
    return DataSampler.generate_samples(structure, num_samples)

# 生成数据并计算统计值
result = generate_test_data(test_structure, 3)
```

## 统计指标说明
- SUM：所有数值型数据的总和
- AVG：所有数值型数据的平均值
- VAR：所有数值型数据的方差
- RMSE：所有数值型数据的均方根误差

## 示例输出
```python
# 生成的样本
{
    'id': 123,
    'scores': [85.5, 92.3, 88.7],
    'nested': {
        'value': 42,
        'array': [15.2, 18.9]
    }
}

# 统计结果
SUM: 456.6
AVG: 76.1
VAR: 1234.5
RMSE: 35.1
```

## 注意事项
- 装饰器会自动收集所有数值型数据（int和float）
- 统计计算会忽略非数值型数据
- 确保数据结构中包含足够的数值型数据以获得有意义的统计结果

## 扩展性
可以通过修改 `StatsDecorator` 类来添加新的统计指标或自定义统计计算方法。 