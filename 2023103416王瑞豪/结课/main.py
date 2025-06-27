from field_rules import FieldRuleParser
from validator import DataValidator
from statistics_decorator import Statistics
from utils import pretty_print, save_to_json
import argparse

# === 1. 定义字段生成规则 ===
RULES = {
    "name": {"type": str, "length": 8},
    "age": {"type": int, "range": (18, 60)},
    "score": {"type": float, "range": (60, 100)},
    "gender": {"type": str, "choices": ["male", "female"]},
    "active": {"type": bool, "probability": 0.7},
    "tags": {"type": list, "subtype": str, "length": 3}
}

# === 2. 定义验证结构（字段:类型） ===
EXPECTED_STRUCTURE = {
    "name": str,
    "age": int,
    "score": float,
    "gender": str,
    "active": bool,
    "tags": list
}

# === 3. 修饰数据生成函数 ===
@Statistics.stat_decorator("SUM", "AVG", "VAR", "RMSE")
def generate_samples(count: int):
    parser = FieldRuleParser(RULES)
    return parser.generate(count)

def main(count: int):
    print(f"\n[★] 正在生成 {count} 条样本数据...")
    result = generate_samples(count)

    samples = result["data"]
    stats = result["stats"]
    type_counts = result["type_counts"]

    print("\n[✓] 样本数据预览（前3条）：")
    pretty_print(samples[:3])

    print("\n[✓] 数值统计信息：")
    pretty_print(stats)

    print("\n[✓] 类型数量统计：")
    pretty_print(type_counts)

    # === 验证样本结构 ===
    validator = DataValidator(EXPECTED_STRUCTURE)
    validation_result = validator.validate_batch(samples)
    validator.report_errors(validation_result)

    # === 保存文件 ===
    save_to_json(samples, "generated_samples.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="嵌套样本数据生成与分析系统")
    parser.add_argument("--count", type=int, default=10, help="生成样本数")
    args = parser.parse_args()

    main(args.count)
