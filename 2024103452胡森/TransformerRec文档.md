# TransformerRec 序列推荐模型文档

## 概述

TransformerRec 是一个基于 Transformer 架构的序列推荐模型，专为顺序推荐任务设计。该模型能够捕捉用户交互序列中的长期依赖关系，为下一个项目预测提供高质量的推荐。

## 主要组件

### 1. FeedForward 模块
- **功能**：Transformer 中的前馈神经网络
- **结构**：
  - 两个线性层
  - GELU 激活函数
  - 残差连接
  - LayerNorm 归一化
- **参数**：
  - `hidden_size`: 隐藏层维度
  - `inner_size`: 内部层维度（通常为隐藏层的4倍）
  - `hidden_dropout_prob`: Dropout 概率
  - `layer_norm_eps`: LayerNorm 的 epsilon 值

### 2. MultiHeadAttention 模块
- **功能**：多头注意力机制
- **结构**：
  - 查询、键、值线性变换
  - 缩放点积注意力
  - Softmax 归一化
  - 残差连接和 LayerNorm
- **参数**：
  - `n_heads`: 注意力头数
  - `hidden_size`: 隐藏层维度
  - `hidden_dropout_prob`: 隐藏层 Dropout 概率
  - `attn_dropout_prob`: 注意力 Dropout 概率
  - `layer_norm_eps`: LayerNorm 的 epsilon 值

### 3. TransformerLayer 模块
- **功能**：Transformer 基础层
- **结构**：
  - 多头注意力子层
  - 前馈网络子层
  - 两层 LayerNorm（分别用于注意力和前馈网络）
- **参数**：同 MultiHeadAttention 和 FeedForward

### 4. TransformerEncoder 模块
- **功能**：Transformer 编码器堆栈
- **结构**：
  - 多个 TransformerLayer 堆叠
  - 最终 LayerNorm 归一化
- **参数**：
  - `n_layers`: Transformer 层数
  - `n_heads`: 注意力头数
  - `hidden_size`: 隐藏层维度
  - `inner_size`: 内部层维度
  - `hidden_dropout_prob`: 隐藏层 Dropout 概率
  - `attn_dropout_prob`: 注意力 Dropout 概率
  - `layer_norm_eps`: LayerNorm 的 epsilon 值

### 5. TransformerRec 模型
- **功能**：完整的序列推荐模型
- **核心组件**：
  - 项目嵌入层
  - 位置嵌入层
  - TransformerEncoder
  - 损失函数（BPR 或交叉熵）
- **参数**：
  - `config`: 模型配置字典
  - `dataset`: 数据集对象

## 关键方法

### 1. 前向传播 (`forward`)
```python
def forward(self, item_seq, item_seq_len)
```
- **输入**：
  - `item_seq`: 项目序列张量 [batch_size, seq_len]
  - `item_seq_len`: 序列实际长度 [batch_size]
- **输出**：
  - 序列表示向量 [batch_size, hidden_size]

### 2. 损失计算 (`calculate_loss`)
```python
def calculate_loss(self, interaction)
```
- **输入**：包含以下键的交互字典
  - `ITEM_SEQ`: 项目序列
  - `ITEM_SEQ_LEN`: 序列长度
  - `POS_ITEM_ID`: 正样本项目ID
  - `NEG_ITEM_ID` (BPR损失时): 负样本项目ID
- **输出**：损失值

### 3. 预测方法
- **单项目预测** (`predict`)
- **全项目排序预测** (`full_sort_predict`)

### 4. 模型保存与加载
```python
def save_model(self, path)
@classmethod
def load_model(cls, path, dataset)
```

### 5. 模型分析工具
```python
count_parameters(model)  # 计算可训练参数数量
model_summary(model)     # 打印模型结构摘要
```

## 使用示例

### 1. 模型初始化
```python
config = {
    "n_layers": 2,
    "n_heads": 2,
    "hidden_size": 64,
    "inner_size": 256,
    "hidden_dropout_prob": 0.5,
    "attn_dropout_prob": 0.5,
    "layer_norm_eps": 1e-12,
    "initializer_range": 0.02,
    "loss_type": "BPR"  # 或 "CE"
}

model = TransformerRec(config, dataset)
```

### 2. 模型训练
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model.calculate_loss(batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

### 3. 模型保存与加载
```python
# 保存模型
model.save_model("transformer_rec.pth")

# 加载模型
loaded_model = TransformerRec.load_model("transformer_rec.pth", dataset)
```

### 4. 模型评估
```python
model.eval()
with torch.no_grad():
    # 全排序预测
    scores = model.full_sort_predict(test_batch)
    # 计算评估指标 (HR, NDCG等)
```

## 配置参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `n_layers` | int | 2 | Transformer层数 |
| `n_heads` | int | 2 | 注意力头数 |
| `hidden_size` | int | 64 | 隐藏层维度 |
| `inner_size` | int | 256 | 前馈网络内部维度 |
| `hidden_dropout_prob` | float | 0.5 | 隐藏层Dropout概率 |
| `attn_dropout_prob` | float | 0.5 | 注意力Dropout概率 |
| `layer_norm_eps` | float | 1e-12 | LayerNorm epsilon值 |
| `initializer_range` | float | 0.02 | 参数初始化范围 |
| `loss_type` | str | "BPR" | 损失函数类型 ("BPR" 或 "CE") |

## 注意事项

1. **数据集要求**：
   - 数据集对象应包含 `num_items`（项目总数）和 `max_seq_length`（最大序列长度）属性
   - 交互数据应包含指定的键：`ITEM_SEQ`, `ITEM_SEQ_LEN`, `POS_ITEM_ID` 等

2. **性能优化**：
   - 对于大型数据集，建议使用混合精度训练
   - 可调整 `hidden_size` 和 `inner_size` 平衡模型容量和计算效率

3. **扩展功能**：
   - 注意力可视化：通过 `get_attention_weights` 方法获取注意力权重
   - 模型分析：使用 `model_summary` 和 `count_parameters` 分析模型结构