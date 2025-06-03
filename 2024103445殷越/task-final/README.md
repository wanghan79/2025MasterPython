# 基于话题分布的影响最大化算法

## 项目概述

本项目实现了一个基于多话题分布的社交网络影响最大化算法。该算法专门针对学术论文引用网络设计，能够根据特定的话题查询选择最优的种子节点集合，以最大化信息传播的覆盖范围。

## 主要目标

1. **多话题影响最大化**：根据用户查询的话题分布，在社交网络中选择最优的种子节点集合
2. **引用网络分析**：处理和分析学术论文的引用关系，构建多话题属性的网络图
3. **算法性能优化**：实现多种优化策略以提高算法效率和效果
4. **实验评估**：对比不同算法的覆盖率和运行时间性能

## 核心技术特点

### 1. 多话题建模
- 每个节点（论文）具有话题分布向量
- 支持多主题数据集的加载和处理
- 基于话题相关性计算边的激活概率

### 2. 算法实现
- **RSS算法**：基于快照的影响最大化算法
- **RSS-v1优化**：使用优先队列优化的边际增益计算
- **度中心性基线**：基于度中心性的对比算法

### 3. 数据处理
- 支持AMiner数据集的加载和解析
- 自动构建多话题属性的引用网络
- 灵活的网络图合并和处理机制

## 文件结构说明

```
task-final/
├── README.md                      # 项目说明文档
├── lfs_aminer_datasets_loader.py  # 数据集加载器
├── im.py                          # 核心影响最大化算法
├── nettools.py                    # 网络工具函数
├── im_exp.py                      # 实验脚本和评估
└── degree_im.py                   # 基于度中心性的基线算法
```

### 文件功能详解

#### `lfs_aminer_datasets_loader.py`
- **主要功能**：加载和处理AMiner学术数据集
- **核心方法**：
  - `read_citation_file()`: 读取单个话题的引用网络文件
  - `read_citation_file_raw()`: 读取原始引用网络数据
  - `combine_topic_graphs()`: 合并多个话题图为统一网络
  - `load_aminer_dataset()`: 完整的数据集加载流程

#### `im.py`
- **主要功能**：实现核心的影响最大化算法
- **核心算法**：
  - `find_seeds_in_candidates()`: RSS算法基础实现
  - `find_seeds_v1()`: RSS-v1优化版本（使用优先队列）
  - `find_seeds_degree()`: 基于度中心性的种子选择
  - `generate_snapshots()`: 生成网络快照用于Monte Carlo模拟

#### `nettools.py`
- **主要功能**：提供网络分析工具函数
- **核心功能**：
  - `monte_carlo_snapshot()`: Monte Carlo信息传播模拟
  - `calculate_coverage()`: 计算种子节点集合的覆盖率

#### `im_exp.py`
- **主要功能**：实验设计和性能评估
- **实验内容**：
  - 生成随机查询和话题分布
  - 对比多种算法的性能
  - 统计覆盖率和运行时间

#### `degree_im.py`
- **主要功能**：基于度中心性的基线算法
- **算法思路**：选择与查询话题相关的加权出度最高的节点

## 实现思路

### 1. 数据预处理阶段
```python
# 加载多话题数据集
graph, num_topics = load_aminer_dataset('./data/lfs.aminer.cn/graphs_pubs')

# 为每个节点添加话题分布向量
add_topic_vector_to_graph(graph, topic_index, num_topics)
```

### 2. 激活概率计算
对于网络中的每条边 (u,v)，计算基于话题分布的激活概率：
```
ap(u,v|γ) = Σ(γ_i × topics_v[i])
```
其中γ是查询的话题分布，topics_v是节点v的话题向量。

### 3. 快照生成
生成R个网络快照，每个快照根据激活概率随机删除边：
```python
snapshots = generate_snapshots(G_gamma, R)
```

### 4. 种子节点选择
使用贪心策略迭代选择k个种子节点，每次选择边际增益最大的节点：
```python
# 计算边际增益
marginal_gain = |reach(S ∪ {v})| - |reach(S)|

# 选择最优节点
best_node = argmax(marginal_gain)
```

### 5. 性能优化策略

#### RSS-v1优化
- 使用优先队列维护候选节点
- 延迟边际增益计算
- 减少重复计算开销

#### 预计算优化
- 预计算每个快照中的可达性信息
- 缓存网络拓扑结构
- 批量处理边的激活概率

## 使用方法

### 基本使用
```python
from im import im
from lfs_aminer_datasets_loader import load_aminer_dataset

# 加载数据集
graph, num_topics = load_aminer_dataset('./data/path')

# 定义查询话题分布
topic_distribution = [0.4, 0.3, 0.3, 0.0, 0.0]  # 示例：关注前3个话题

# 执行影响最大化
seeds_rss, seeds_v1, seeds_degree = im(graph, num_topics, topic_distribution, k=10)
```

### 实验评估
```python
from im_exp import experiment_with_im

# 执行完整实验
results = experiment_with_im(graph, num_topics, R=100)

# 查看结果
print("RSS算法覆盖率:", results['IM']['coverage'])
print("RSS-v1算法覆盖率:", results['IM-v1']['coverage'])
```

## 算法复杂度

- **时间复杂度**：O(k × R × |V| × |E|)
  - k: 种子节点数量
  - R: 快照数量
  - |V|: 节点数量
  - |E|: 边数量

- **空间复杂度**：O(R × |V|²)
  - 主要用于存储预计算的可达性信息

## 实验结果

项目支持以下性能指标的评估：
- **覆盖率**：种子节点集合能够影响的平均节点数量
- **运行时间**：算法执行的平均时间
- **扩展性**：不同参数设置下的性能表现

## 依赖环境

```bash
pip install networkx numpy tqdm
```

## 数据格式

支持的数据格式为AMiner引用网络格式：
```
*Vertices n
node_id "title" cited_count
...
*Edges
source_id target_id weight
...
```

## 扩展性

该项目具有良好的扩展性，支持：
- 自定义话题分布计算方法
- 不同的网络快照生成策略
- 多种种子节点选择算法
- 灵活的实验参数配置

## 贡献

本项目实现了基于多话题分布的影响最大化算法，为学术网络分析和信息传播研究提供了有效的工具和方法。 