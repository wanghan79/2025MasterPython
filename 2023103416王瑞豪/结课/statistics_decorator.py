from functools import wraps
from typing import Callable, Any
import math

class StatisticsDecorator:
    def __init__(self, *stats: str):
        self.stats = set(stat.upper() for stat in stats)

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> dict:
            result = func(*args, **kwargs)
            numeric_values = self._extract_numbers(result)
            stats_result = {}

            if not numeric_values:
                return {"stats": {}, "data": result}

            if "SUM" in self.stats:
                stats_result["SUM"] = sum(numeric_values)
            if "AVG" in self.stats:
                stats_result["AVG"] = sum(numeric_values) / len(numeric_values)
            if "VAR" in self.stats:
                mean = sum(numeric_values) / len(numeric_values)
                stats_result["VAR"] = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
            if "RMSE" in self.stats:
                mean = sum(numeric_values) / len(numeric_values)
                mse = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
                stats_result["RMSE"] = math.sqrt(mse)

            return {
                "stats": stats_result,
                "data": result
            }

        return wrapper

    def _extract_numbers(self, data: Any) -> list[float]:
        numbers = []
        if isinstance(data, (int, float)):
            numbers.append(data)
        elif isinstance(data, (list, tuple, set)):
            for item in data:
                numbers.extend(self._extract_numbers(item))
        elif isinstance(data, dict):
            for val in data.values():
                numbers.extend(self._extract_numbers(val))
        return numbers