import random
import string
from typing import Any, Dict, List, Tuple, Union

class FieldRuleParser:
    def __init__(self, rule_dict: Dict[str, Dict[str, Any]]):
        self.rules = rule_dict

    def generate_field(self, rule: Dict[str, Any]) -> Any:
        field_type = rule.get("type")

        if field_type == int:
            low, high = rule.get("range", (0, 100))
            return random.randint(low, high)

        elif field_type == float:
            low, high = rule.get("range", (0.0, 100.0))
            return round(random.uniform(low, high), 2)

        elif field_type == str:
            if "choices" in rule:
                return random.choice(rule["choices"])
            length = rule.get("length", 6)
            return ''.join(random.choices(string.ascii_letters, k=length))

        elif field_type == bool:
            p = rule.get("probability", 0.5)
            return random.random() < p

        elif field_type == list:
            subtype = rule.get("subtype", int)
            length = rule.get("length", 3)
            sub_rule = {"type": subtype}
            if "range" in rule:
                sub_rule["range"] = rule["range"]
            return [self.generate_field(sub_rule) for _ in range(length)]

        else:
            return None

    def generate_one(self) -> Dict[str, Any]:
        return {field: self.generate_field(rule) for field, rule in self.rules.items()}

    def generate(self, count: int) -> List[Dict[str, Any]]:
        return [self.generate_one() for _ in range(count)]


# 示例调用
if __name__ == "__main__":
    rule_config = {
        "age": {"type": int, "range": (18, 60)},
        "score": {"type": float, "range": (0, 100)},
        "name": {"type": str, "length": 10},
        "gender": {"type": str, "choices": ["男", "女"]},
        "active": {"type": bool, "probability": 0.7},
        "tags": {"type": list, "subtype": str, "length": 4}
    }

    parser = FieldRuleParser(rule_config)
    samples = parser.generate(5)

    from utils import pretty_print
    pretty_print(samples)
