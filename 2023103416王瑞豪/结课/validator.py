from typing import Any, Dict, List, Tuple, Union

class DataValidator:
    def __init__(self, expected_structure: Dict[str, type]):
        """
        expected_structure: {'name': str, 'age': int, ...}
        """
        self.expected = expected_structure

    def check_type_match(self, value: Any, expected_type: type) -> bool:
        if expected_type == float:
            return isinstance(value, (float, int))  # int 可以作为 float 使用
        return isinstance(value, expected_type)

    def validate_sample(self, sample: Dict[str, Any]) -> List[str]:
        errors = []

        for field, expected_type in self.expected.items():
            if field not in sample:
                errors.append(f"缺失字段: '{field}'")
                continue

            actual_value = sample[field]

            if not self.check_type_match(actual_value, expected_type):
                errors.append(
                    f"字段类型错误: '{field}' 应为 {expected_type.__name__}，实际为 {type(actual_value).__name__}"
                )

        # 检查是否有多余字段
        for field in sample.keys():
            if field not in self.expected:
                errors.append(f"未知字段: '{field}' 不在预期结构中")

        return errors

    def validate_batch(self, samples: List[Dict[str, Any]]) -> List[Tuple[int, List[str]]]:
        """
        批量验证所有样本
        返回：(样本索引, 错误列表)
        """
        results = []
        for i, sample in enumerate(samples):
            errs = self.validate_sample(sample)
            if errs:
                results.append((i, errs))
        return results

    def report_errors(self, batch_errors: List[Tuple[int, List[str]]]) -> None:
        print("\n=== 验证错误报告 ===")
        for index, errs in batch_errors:
            print(f"[样本 #{index}]")
            for err in errs:
                print("  -", err)
        if not batch_errors:
            print("✓ 所有样本通过验证")
