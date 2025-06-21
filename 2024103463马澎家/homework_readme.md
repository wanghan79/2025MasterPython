# 作业1：Python 容器性能测试：List vs Tuple

## 实验目的

本实验旨在比较 Python 中**可变容器 `list`** 与**不可变容器 `tuple`**在大规模数据修改操作中的性能差异。

---

## 实验设置

- 构建两个 `10000 × 10000` 的矩阵：
  - 一个由 `list` 构成（可变）
  - 一个由 `tuple` 构成（不可变）

- 执行 10000 次随机位置的元素修改：
  - 对于 `list`：直接修改指定位置的元素
  - 对于 `tuple`：由于其不可变性，每次修改都需要：
    1. 将目标行转换为 `list`
    2. 修改元素
    3. 再转换回 `tuple`
    4. 重建整行并替换入矩阵

---

## ⏱ 输出结果
构造 list 矩阵...
开始修改 list 中的元素...
list 修改耗时：0.02 秒

构造 tuple 矩阵...
开始修改 tuple 中的元素（通过重建）...
tuple 修改耗时：5.03 秒

## 实验结论
在大量频繁修改数据的场景中，应优先使用 list；而 tuple 更适用于只读和作为键的用途（如 dict 的 key）。


# 作业二：结构化随机数据生成器（DataSampler）

## 项目简介

本项目实现了一个通用的结构化随机数据生成器 `DataSampler`，支持任意嵌套的数据结构和多种基本数据类型的随机生成，适用于模拟用户信息、接口测试数据、数据分析样本等场景。

---

## 功能特点

- 支持嵌套结构：可递归生成嵌套的 `dict`、`list`、`tuple` 等结构
- 多类型数据生成：包括 `int`、`float`、`str`、`bool`、`date` 等常见类型
- 动态结构定义：通过 `kwargs` 自定义结构和字段类型
- 灵活样本数量：通过参数控制生成的样本数量

---

## 支持的数据类型

| 类型     | 用法                      | 说明                                  |
|----------|---------------------------|---------------------------------------|
| int      | `"int"`                  | 随机整数（0 ~ 100）                   |
| float    | `"float"`                | 随机浮点数（0.00 ~ 100.00）           |
| str      | `"str"`                  | 随机 8 位字母数字字符串               |
| bool     | `"bool"`                 | 随机布尔值 `True` 或 `False`         |
| date     | `"date"`                 | 随机日期（2000~2025 年间）           |
| list     | `['list', 类型, 数量]`   | 指定类型的列表（支持嵌套结构）       |
| tuple    | `['tuple', 元素类型...]` | 固定结构元组（可嵌套多种类型）       |
| dict     | `{key: type}`            | 任意深度嵌套的键值对结构              |

---

## 使用示例

```python
from data_sampler import data_sampler

schema = {
    "id": "int",
    "name": "str",
    "signup": "date",
    "active": "bool",
    "profile": {
        "age": "int",
        "score": "float",
        "tags": ["list", "str", 3],
        "location": {
            "lat": "float",
            "lng": "float"
        }
    },
    "history": ["list", {"time": "date", "action": "str"}, 2],
    "device": ["tuple", "str", "bool", "float"]
}

# 生成 3 条样本
samples = data_sampler(3, **schema)

for s in samples:
    print(s)
```

# 作业三：结构化数据统计修饰器（DataSampler + Stats）

## 项目简介

本项目在作业二的结构化随机数据生成器基础上，使用带参数的 Python 修饰器 `@stats_decorator` 实现对生成数据的自动数值统计分析。最终可输出多个样本中所有数值型叶子节点的统计特征，包括：

- SUM（求和）
- AVG（平均值）
- VAR（方差）
- RMSE（均方根误差）

---

## 实验目的

- 掌握 Python 修饰器的编写方法，尤其是**带参数修饰器**
- 掌握如何在复杂嵌套结构中提取数值信息进行统计
- 实现对结构化样本的数值型字段的自动分析

---

## 功能组成

| 模块 | 功能说明 |
|------|----------|
| `data_sampler` | 结构化随机样本生成器（支持嵌套结构、多个数据类型） |
| `@stats_decorator(...)` | 带参数的修饰器，可对生成数据自动进行统计 |
| `extract_numeric_values` | 递归提取嵌套数据结构中的所有数值型叶子节点 |
| `main()` | 示例入口，运行生成与统计过程 |

---

## 支持数据类型

| 类型     | 用法示例              | 说明                                |
|----------|-----------------------|-------------------------------------|
| `int`    | `"int"`               | 随机整数（0~100）                  |
| `float`  | `"float"`             | 随机浮点数（0.00~100.00）          |
| `str`    | `"str"`               | 随机 8 位字符串                    |
| `bool`   | `"bool"`              | 布尔值                              |
| `date`   | `"date"`              | 随机日期（2000~2025）              |
| `list`   | `["list", type, N]`   | 包含 N 个指定类型的列表            |
| `tuple`  | `["tuple", ...types]` | 包含多个不同字段的固定结构元组     |
| `dict`   | `{key: type}`         | 支持任意深度嵌套的结构             |

---

## 使用示例

### 调用方式

```python
@stats_decorator("SUM", "AVG", "VAR", "RMSE")
def generate_and_analyze():
    return data_sampler(5, **schema)

result = generate_and_analyze()
print(result["stats"])
