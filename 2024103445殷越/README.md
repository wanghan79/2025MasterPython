# 文件介绍

## 第一次作业：`matrix_performance_test.py`

这个文件是一个性能测试脚本，用于比较列表和元组在矩阵操作中的性能差异：

- 创建大型矩阵并进行随机元素修改
- 测量不同数据结构(列表 vs 元组)的操作时间
- 分析性能差异并输出比较结果

## 第二次作业：`data_sampling.py`

这个文件提供了基础的随机数据生成功能。主要特点：

- `create_random_object()` 函数：根据指定的类型生成随机对象
- 支持生成多种数据类型：整数、浮点数、字符串、布尔值以及嵌套对象
- 可以生成复杂的数据结构，如嵌套字典、用户自定义类的实例等

## 第三次作业（基础版）：`data_sampling_decorators.py`

这个文件提供了统计数据分析的装饰器。主要功能：

- 定义了 `statistical_decorator` 装饰器，可以对随机生成的数据进行统计分析
- 支持多种统计方法：均值、方差、均方根误差(RMSE)和求和
- 提供了 `sample_generator` 和 `extract_field_values` 辅助函数

## 第三次作业（可选改进版）：`data_sampling_decorators_v2.py`

这是对 `data_sampling_decorators.py` 的改进版本，采用了面向对象的设计方法：

- 定义了 `StatisticalMethod` 抽象基类及其具体实现类
- 实现了工厂模式 `StatisticalMethodFactory` 来创建不同的统计方法
- 优化了代码结构，增强了可扩展性

## 期末作业：`task-final` 文件夹

是对某个影响力最大化算法的实现尝试。

## `test01.py`

这个文件包含了多个Python对象和数据类型的测试用例，主要关注：

- 各种数据类型的内存地址变化情况（`id()`函数）
- 可变类型和不可变类型的行为差异
- `copy`和`deepcopy`的使用和比较
- 混合数据类型容器的特性
