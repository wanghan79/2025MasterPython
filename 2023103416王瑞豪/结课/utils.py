import json
import os
from typing import Any, List
from pprint import pprint

def save_to_json(data: Any, filename: str = "output.json") -> None:
    """
    将数据保存为 JSON 文件
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[✓] 数据已保存到 {os.path.abspath(filename)}")

def pretty_print(data: Any) -> None:
    """
    使用 pprint 格式化打印数据
    """
    print("========== Pretty Print ==========")
    pprint(data, indent=2, width=80)

def flatten_numbers(data: Any) -> List[float]:
    """
    提取嵌套结构中所有 int / float 数值
    """
    result = []

    if isinstance(data, (int, float)):
        result.append(data)
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            result.extend(flatten_numbers(item))
    elif isinstance(data, dict):
        for val in data.values():
            result.extend(flatten_numbers(val))

    return result

def structure_summary(data: Any, level=0) -> dict:
    """
    对结构进行分析，返回字段数量、最大嵌套层级等信息
    """
    if not isinstance(data, (dict, list, tuple)):
        return {"fields": 1, "max_depth": level}

    fields = 0
    max_depth = level

    if isinstance(data, dict):
        for v in data.values():
            sub_summary = structure_summary(v, level + 1)
            fields += sub_summary["fields"]
            max_depth = max(max_depth, sub_summary["max_depth"])
    elif isinstance(data, (list, tuple)):
        for item in data:
            sub_summary = structure_summary(item, level + 1)
            fields += sub_summary["fields"]
            max_depth = max(max_depth, sub_summary["max_depth"])

    return {"fields": fields, "max_depth": max_depth}
