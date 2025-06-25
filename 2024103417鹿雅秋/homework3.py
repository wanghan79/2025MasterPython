import random
import string
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable, Set, Union, Tuple
from functools import wraps
import math

class DataSampler:
    def __init__(self):
        self.type_generators = {
            'int': self._generate_int,
            'float': self._generate_float,
            'str': self._generate_str,
            'bool': self._generate_bool,
            'date': self._generate_date,
            'list': self._generate_list,
            'tuple': self._generate_tuple,
            'dict': self._generate_dict
        }

    def _generate_int(self, min_val: int = 0, max_val: int = 100) -> int:
        return random.randint(min_val, max_val)

    def _generate_float(self, min_val: float = 0.0, max_val: float = 100.0) -> float:
        return random.uniform(min_val, max_val)

    def _generate_str(self, length: int = 10) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def _generate_bool(self) -> bool:
        return random.choice([True, False])

    def _generate_date(self, start_year: int = 2000, end_year: int = 2024) -> str:
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime('%Y-%m-%d')

    def _generate_list(self, schema: Dict[str, Any], length: int = 3) -> List[Any]:
        return [self.generate_sample(schema) for _ in range(length)]

    def _generate_tuple(self, schema: Dict[str, Any], length: int = 3) -> Tuple[Any, ...]:
        return tuple(self.generate_sample(schema) for _ in range(length))

    def _generate_dict(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self.generate_sample(value) for key, value in schema.items()}

    def generate_sample(self, schema: Union[Dict[str, Any], str, List[Any]]) -> Any:
        if isinstance(schema, str):
            # 处理基本数据类型
            if schema in self.type_generators:
                return self.type_generators[schema]()
        elif isinstance(schema, dict):
            if 'type' in schema:
                # 处理带有特定参数的数据类型
                data_type = schema['type']
                if data_type in self.type_generators:
                    params = {k: v for k, v in schema.items() if k != 'type'}
                    return self.type_generators[data_type](**params)
            else:
                # 处理嵌套字典
                return self._generate_dict(schema)
        elif isinstance(schema, list):
            # 处理列表类型
            return [self.generate_sample(item) for item in schema]
        raise ValueError(f"Unsupported schema type: {schema}")

    def generate_samples(self, schema: Dict[str, Any], num_samples: int = 1) -> List[Any]:
        """生成多个数据样本"""
        return [self.generate_sample(schema) for _ in range(num_samples)]

class StatsDecorator:
    """统计分析装饰器类"""
    
    def __init__(self, stats: Union[str, List[str], Set[str]] = None):
        """
        初始化装饰器
        :param stats: 需要计算的统计指标，可选：'SUM', 'AVG', 'VAR', 'RMSE'
        """
        if stats is None:
            stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
        elif isinstance(stats, str):
            stats = {stats}
        elif isinstance(stats, list):
            stats = set(stats)
            
        self.stats = stats
        self._validate_stats()
    
    def _validate_stats(self):
        """验证统计指标的有效性"""
        valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
        invalid_stats = self.stats - valid_stats
        if invalid_stats:
            raise ValueError(f"不支持的统计指标: {invalid_stats}. 支持的指标: {valid_stats}")
    
    def __call__(self, func: Callable) -> Callable:
        """装饰器调用方法"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Dict[str, float]]:
            # 获取原始数据
            samples = func(*args, **kwargs)
            if not isinstance(samples, list):
                samples = [samples]
            
            # 分析数据
            return self.analyze(samples)
        
        return wrapper
    
    def _extract_numeric_values(self, data: Any, path: str = '') -> Dict[str, List[float]]:
        """递归提取所有数值型叶节点"""
        numeric_values = {}
        
        if isinstance(data, (int, float)):
            numeric_values[path] = [float(data)]
        elif isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                nested_values = self._extract_numeric_values(value, new_path)
                numeric_values.update(nested_values)
        elif isinstance(data, (list, tuple)):
            for i, value in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                nested_values = self._extract_numeric_values(value, new_path)
                numeric_values.update(nested_values)
        
        return numeric_values
    
    def analyze(self, samples: List[Any]) -> Dict[str, Dict[str, float]]:
        """分析数据中的数值型叶节点，返回统计结果"""
        # 提取所有样本中的数值
        all_numeric_values: Dict[str, List[float]] = {}
        for sample in samples:
            sample_values = self._extract_numeric_values(sample)
            for path, values in sample_values.items():
                if path not in all_numeric_values:
                    all_numeric_values[path] = []
                all_numeric_values[path].extend(values)
        
        # 计算统计指标
        results = {}
        for path, values in all_numeric_values.items():
            results[path] = self._calculate_stats(values)
        
        return results
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """计算指定的统计指标"""
        stats_results = {}
        n = len(values)
        
        if 'SUM' in self.stats:
            stats_results['SUM'] = sum(values)
        
        if 'AVG' in self.stats:
            stats_results['AVG'] = sum(values) / n
        
        if 'VAR' in self.stats or 'RMSE' in self.stats:
            mean = sum(values) / n
            squared_diff_sum = sum((x - mean) ** 2 for x in values)
            
            if 'VAR' in self.stats:
                stats_results['VAR'] = squared_diff_sum / n
            
            if 'RMSE' in self.stats:
                stats_results['RMSE'] = math.sqrt(squared_diff_sum / n)
        
        return stats_results

def main():
    # 创建数据采样器
    sampler = DataSampler()
    
    # 定义测试数据模式
    schema = {
        "user": {
            "id": "int",
            "age": {"type": "int", "min_val": 18, "max_val": 80},
            "scores": {"type": "list", "schema": {"type": "float", "min_val": 60, "max_val": 100}, "length": 3}
        },
        "metrics": {
            "daily_usage": {"type": "float", "min_val": 0, "max_val": 24},
            "performance": {"type": "list", "schema": "float", "length": 2}
        }
    }
    
    # 使用不同的统计指标组合测试装饰器
    @StatsDecorator(['AVG', 'VAR'])
    def generate_data_with_avg_var(schema, num_samples):
        return sampler.generate_samples(schema, num_samples)
    
    @StatsDecorator(['SUM', 'RMSE'])
    def generate_data_with_sum_rmse(schema, num_samples):
        return sampler.generate_samples(schema, num_samples)
    
    @StatsDecorator()  # 使用所有统计指标
    def generate_data_with_all_stats(schema, num_samples):
        return sampler.generate_samples(schema, num_samples)
    
    # 生成并分析数据
    print("1. 均值和方差统计：")
    avg_var_stats = generate_data_with_avg_var(schema, 5)
    print(avg_var_stats)
    
    print("\n2. 求和和均方根误差统计：")
    sum_rmse_stats = generate_data_with_sum_rmse(schema, 5)
    print(sum_rmse_stats)
    
    print("\n3. 所有统计指标：")
    all_stats = generate_data_with_all_stats(schema, 5)
    print(all_stats)

if __name__ == "__main__":
    main()
