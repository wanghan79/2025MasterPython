import random
from typing import Any, Dict, List, Tuple, Union


class DataGenerator:
    """
    结构化随机数据生成器，支持嵌套数据结构和多种随机值生成策略
    """

    @staticmethod
    def generate(num: int, data_template: Dict[str, Any], add_index: bool = True) -> List[Dict[str, Any]]:
        """
        基于模板生成指定数量的随机数据

        参数:
            num: 生成的数据数量
            data_template: 数据结构模板，支持嵌套字典
            add_index: 是否在顶级键后添加索引编号

        返回:
            生成的随机数据列表
        """
        if not isinstance(num, int) or num <= 0:
            raise ValueError("数据数量必须是正整数")

        result = []
        for i in range(num):
            element = {}
            for key, value in data_template.items():
                processed_value = DataGenerator._process_value(value)
                element[f"{key}{i}"] = processed_value if add_index else processed_value
            result.append(element)
        return result

    @staticmethod
    def _process_value(data: Any) -> Any:
        """
        递归处理数据结构，根据类型生成随机值

        支持的特殊结构:
            - (min, max) 形式的元组/列表: 生成范围内的随机数
            - 字典: 递归处理每个值
            - 其他类型: 直接返回
        """
        if isinstance(data, dict):
            return {k: DataGenerator._process_value(v) for k, v in data.items()}

        if isinstance(data, (list, tuple)) and len(data) == 2:
            if all(isinstance(x, int) for x in data):
                return random.randint(data[0], data[1])
            if all(isinstance(x, float) for x in data):
                return round(random.uniform(data[0], data[1]), 2)
            if isinstance(data[0], str) and isinstance(data[1], int):
                return DataGenerator._generate_random_string(data[0], data[1])

        return data

    @staticmethod
    def _generate_random_string(char_set: str, length: int) -> str:
        """生成指定字符集和长度的随机字符串"""
        return ''.join(random.choice(char_set) for _ in range(length))


# 示例用法
if __name__ == "__main__":
    # 定义数据模板
    TOWN_TEMPLATE = {
        "town": {
            "school": {
                "teachers": (50, 70),
                "students": (800, 1200),
                "others": (20, 40),
                "budget": (410000.5, 986553.1),
                "code": ("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", 6)  # 随机字符串示例
            },
            "hospital": {
                "doctors": (40, 60),
                "nurses": (60, 80),
                "patients": (200, 300),
                "revenue": (110050.5, 426553.4)
            },
            "supermarket": {
                "cashiers": (10, 30),
                "shelves": (50, 100),
                "daily_customers": (500, 1500),
                "annual_profit": (310000.3, 7965453.4)
            }
        }
    }

    # 生成并打印数据
    generator = DataGenerator()
    generated_data = generator.generate(3, TOWN_TEMPLATE)

    for idx, data in enumerate(generated_data):
        print(f"\n数据组 {idx + 1}:")
        for key, value in data.items():
            print(f"  {key}: {value}")