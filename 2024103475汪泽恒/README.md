
## 作业一：Python数据固定值和可变值容器性能测试

### 功能描述
- 比较Python中可变数据结构(list)和不可变数据结构(tuple)在修改操作上的性能差异
- 创建10000×10000的数据矩阵，分别进行10000轮修改，记录时间消耗

### 实现原理
- 列表(list)是可变的，可以直接修改元素值
- 元组(tuple)是不可变的，每次"修改"都需要创建新的元组，因此效率极低
- 通过随机选择位置进行修改，测量修改过程的时间消耗

### 使用方法
```bash
python assignment1.py
```

### 预期结果
程序将显示列表和元组修改操作的时间消耗对比，并给出性能差异结论。由于元组修改成本极高，实际测试使用了较小的元组矩阵，但仍能反映出性能差异。

## 作业二：随机数据生成器(DataSampler)

### 功能描述
- 实现一个灵活的随机数据生成器，支持生成结构化的模拟数据
- 支持多种数据类型：int、float、str、bool、date、list、tuple、dict
- 支持任意嵌套的数据结构定义和生成

### 实现原理
- 使用类封装各种数据类型的生成方法
- 通过递归生成嵌套数据结构
- 采用schema定义方式描述数据结构和生成规则

### 使用方法
```bash
python assignment2.py
```

### 示例
```python
from assignment2 import DataSampler

# 实例化数据生成器
sampler = DataSampler()

# 定义数据结构模式
user_schema = {
    "id": {"type": "int", "min": 1000, "max": 9999},
    "name": {"type": "str", "length": 8},
    "is_active": {"type": "bool"},
    "scores": {"type": "list", "item_type": {"type": "int", "min": 0, "max": 100}, "length": 3}
}

# 生成样本
users = sampler.generate_samples(user_schema, 5)  # 生成5个样本
```

## 作业三：数据统计修饰器

### 功能描述
- 使用Python修饰器计算数据的统计特征
- 支持多种统计操作：SUM（求和）、AVG（均值）、VAR（方差）、RMSE（均方根误差）
- 可以灵活组合使用多种统计操作

### 实现原理
- 实现一个带参数的修饰器，接收要执行的统计操作
- 修饰器包装原始函数，在返回数据的同时计算统计结果
- 递归遍历数据结构提取所有数值型叶节点进行统计

### 使用方法
```bash
python assignment3.py
```

### 示例
```python
from assignment2 import DataSampler
from assignment3 import stats_decorator

# 实例化数据生成器
sampler = DataSampler()

# 定义数据结构
schema = {...}  # 数据结构定义

# 使用修饰器
@stats_decorator("SUM", "AVG", "VAR", "RMSE")
def generate_data(sampler, schema, count):
    return sampler.generate_samples(schema, count)

# 调用函数并获取结果
result = generate_data(sampler, schema, 10)
print("原始数据:", result["data"])
print("统计结果:", result["stats"])
```

## 依赖库
- Python 3.6+
- numpy
- datetime 