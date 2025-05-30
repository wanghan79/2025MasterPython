# Python编程作业要求


## 第一次作业：可变与不可变数据结构性能对比

### 作业目标
比较Python中可变数据结构（列表）和不可变数据结构（元组）在修改操作上的性能差异。

### 具体要求
1. 创建两个大型二维矩阵：一个使用嵌套列表，另一个使用嵌套元组
2. 对两种矩阵进行多次随机位置的修改操作
3. 记录并比较两种数据结构修改操作所需的时间
4. 分析结果并解释性能差异的原因

### 实现提示
- 使用`time`模块记录操作时间
- 使用`random`模块生成随机位置
- 列表可以直接修改元素值，而元组需要创建新的元组（因为元组是不可变的）
- 建议使用足够大的矩阵和足够多的修改次数以观察明显的性能差异

### 预期结果
列表的修改操作应显著快于元组，因为元组的每次修改都需要创建新的元组对象。

## 第二次作业：灵活的数据采样器

### 作业目标
实现一个灵活的数据采样器，能够根据指定的数据结构生成随机样本数据。

### 具体要求
1. 设计并实现一个`DataSampler`类，支持以下功能：
   - 支持多种数据类型的随机生成（整数、浮点数、字符串、布尔值、列表、元组、字典、日期等）
   - 支持指定数值范围、字符串长度、列表/元组长度等参数
   - 支持嵌套的复杂数据结构

2. 实现方法：
   - 初始化方法，可选择性地接收随机数生成器
   - 随机值生成方法，根据指定的数据类型和参数生成随机值
   - 结构生成方法，根据指定的结构生成完整的数据样本

3. 提供示例，展示如何使用该采样器生成复杂的数据结构

### 实现提示
- 使用`random`模块生成随机数
- 使用`string`模块提供字符集
- 使用`datetime`模块处理日期
- 采用递归方法处理嵌套结构

### 预期结果
能够根据指定的结构定义生成符合要求的随机数据样本。

## 第三次作业：装饰器与数据分析

### 作业目标
在第二次作业的基础上，使用装饰器模式增加数据分析功能。

### 具体要求
1. 实现一个统计装饰器`stats_decorator`，支持以下统计指标的计算：
   - 均值（mean）
   - 方差（variance）
   - 均方根误差（rmse）
   - 总和（sum）

2. 增强`DataSampler`类：
   - 添加`get_leaf_nodes`方法，用于提取数据结构中的所有叶节点
   - 添加`analyze`方法，使用装饰器计算统计指标

3. 提供示例，展示如何生成随机数据并进行统计分析

### 实现提示
- 使用`functools.wraps`保留被装饰函数的元数据
- 使用`numpy`库进行统计计算
- 采用递归方法遍历数据结构获取叶节点
- 只对数值型（整数和浮点数）叶节点进行统计

### 预期结果
能够生成随机数据样本，并自动计算样本中所有数值型数据的统计指标。
