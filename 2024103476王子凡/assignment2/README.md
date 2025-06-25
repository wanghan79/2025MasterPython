# Python随机数据生成器

## 项目说明
本项目实现了一个灵活的随机数据生成器（DataSampler），可以生成任意嵌套结构的随机数据样本。支持多种数据类型的随机生成，包括基本类型和复杂嵌套结构。

## 功能特点
- 支持多种数据类型的随机生成：
  - 整数（int）
  - 浮点数（float）
  - 字符串（str）
  - 布尔值（bool）
  - 日期时间（datetime）
  - 列表（list）
  - 元组（tuple）
  - 字典（dict）
- 支持任意深度的嵌套数据结构
- 可自定义生成样本数量
- 完全可配置的数据结构模板

## 运行环境要求
- Python 3.6+
- numpy

## 安装依赖
```bash
pip install numpy
```

## 使用方法
1. 定义数据结构模板：
```python
user_structure = {
    "id": int,
    "name": str,
    "age": int,
    "is_active": bool,
    "scores": [float, float, float],
    "address": {
        "city": str,
        "zip_code": str
    },
    "created_at": datetime
}
```

2. 生成随机样本：
```python
from data_sampler import DataSampler

# 生成3个样本
samples = DataSampler.generate_samples(user_structure, 3)
```

## 示例代码
```python
# 生成嵌套列表数据
nested_list_structure = [
    [int, float],
    [str, bool],
    {"key1": int, "key2": [float, float]}
]

# 生成2个样本
samples = DataSampler.generate_samples(nested_list_structure, 2)
```

## 注意事项
- 确保定义的数据结构模板是有效的
- 对于复杂嵌套结构，生成大量样本可能需要较长时间
- 随机生成的数据仅用于测试目的

## 扩展性
可以通过修改 `_generate_value` 方法来支持更多数据类型的随机生成。 