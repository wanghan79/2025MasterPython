from data_sampler import DataSampler
from stat_analyzer import analyze

# 1 结构定义
structure = {
    "id": int,
    "score": float,
    "name": str,
    "active": bool
}

# 2 生成数据
sampler = DataSampler(structure)
data = sampler.sample(20)

# 3 分析
scores = [record["score"] for record in data]
print("原始 scores：", scores)

# print结果
result = analyze(scores)
print("统计分析结果：", result)

