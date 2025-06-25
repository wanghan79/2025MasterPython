import pandas as pd

# 加载 CSV 文件
csv_path = "/root/fssd/SMFFDDG/data/test/S669_with_ddg.csv"
df = pd.read_csv(csv_path)

# 检查 ddg 列是否存在缺失值
missing_ddg = df['ddg'].isna()

# 统计缺失数量
num_missing = missing_ddg.sum()
print(f"⚠️ 共有 {num_missing} 条记录的 ddg 值缺失。")

# 如果存在缺失，打印前几行查看
if num_missing > 0:
    print("\n以下是部分 ddg 缺失的记录：")
    print(df[missing_ddg].head())
else:
    print("✅ 所有记录的 ddg 值均不为空。")
