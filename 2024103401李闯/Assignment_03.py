import random
import string
import math
from datetime import datetime, time, timedelta
from typing import Any, Union, Dict, List, Tuple, Callable, Set, Optional
from functools import wraps


def stats_decorator(stats: List[str] = None):
    """带参数的装饰器，用于统计样本集中的数值型数据
    
    :param stats: 统计项列表，可选值: 'SUM', 'AVG', 'VAR', 'RMSE'
    :return: 装饰器函数
    """
    # 设置默认统计项
    if stats is None:
        stats = ['SUM', 'AVG', 'VAR', 'RMSE']
    else:
        # 确保统计项大写
        stats = [s.upper() for s in stats]
    
    # 验证统计项
    valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
    invalid_stats = set(stats) - valid_stats
    if invalid_stats:
        raise ValueError(f"Invalid stats: {', '.join(invalid_stats)}. "
                         f"Valid stats are: {', '.join(valid_stats)}")
    
    def decorator(func: Callable) -> Callable:
        """实际的装饰器函数
        
        :param func: 被装饰的函数（通常是generate方法）
        :return: 包装后的函数
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Dict[str, Any]:
            """包装函数，添加统计功能
            
            :return: 包含样本集和统计结果的字典
            """
            # 调用原始函数生成样本
            samples = func(self, *args, **kwargs)
            
            # 初始化数值字段收集器
            numeric_fields: Dict[str, List[Union[int, float]]] = {}
            
            # 遍历所有样本收集数值字段
            for sample in samples:
                for key, value in sample.items():
                    # 只收集数值类型（int和float）
                    if isinstance(value, (int, float)):
                        if key not in numeric_fields:
                            numeric_fields[key] = []
                        numeric_fields[key].append(value)
            
            # 计算统计结果
            statistics: Dict[str, Dict[str, Optional[float]]] = {}
            for field, values in numeric_fields.items():
                n = len(values)
                if n == 0:
                    continue
                
                # 计算各项统计指标
                field_stats: Dict[str, Optional[float]] = {}
                
                # 求和
                if 'SUM' in stats:
                    field_stats['SUM'] = sum(values)
                
                # 平均值
                if 'AVG' in stats:
                    field_stats['AVG'] = sum(values) / n
                
                # 方差和标准差
                if 'VAR' in stats or 'RMSE' in stats:
                    mean = field_stats['AVG'] if 'AVG' in field_stats else sum(values) / n
                    variance = sum((x - mean) ** 2 for x in values) / n
                    
                    if 'VAR' in stats:
                        field_stats['VAR'] = variance
                    
                    if 'RMSE' in stats:
                        field_stats['RMSE'] = math.sqrt(variance)
                
                statistics[field] = field_stats
            
            return {
                'samples': samples,
                'statistics': statistics
            }
        
        return wrapper
    
    return decorator


class DataSampler:
    def __init__(self, seed: int = None):
        """Initialize the random data generator with an optional seed."""
        self.random = random.Random(seed)
    
    @stats_decorator(stats=['SUM', 'AVG', 'VAR', 'RMSE'])  # 应用装饰器
    def generate(self, **kwargs) -> List[Dict[str, Any]]:
        """Generate structured random data samples based on the schema."""
        num_samples = kwargs.pop('num_samples', 1)
        samples = []
        for _ in range(num_samples):
            sample = {}
            for field, field_spec in kwargs.items():
                sample[field] = self._generate_field(field_spec)
            samples.append(sample)
        return samples
    
    def _generate_field(self, field_spec: Any) -> Any:
        """Generate a random value based on the field specification."""
        if not isinstance(field_spec, dict):
            return field_spec
        
        data_type = field_spec.get('type')
        
        if data_type == 'int':
            return self._generate_int(field_spec)
        elif data_type == 'float':
            return self._generate_float(field_spec)
        elif data_type == 'str':
            return self._generate_str(field_spec)
        elif data_type == 'bool':
            return self.random.choice([True, False])
        elif data_type == 'date':
            return self._generate_date(field_spec)
        elif data_type == 'list':
            return self._generate_list(field_spec)
        elif data_type == 'tuple':
            return self._generate_tuple(field_spec)
        elif data_type == 'dict':
            return self._generate_dict(field_spec)
        elif data_type == 'choice':
            return self._generate_choice(field_spec)
        elif data_type == 'datetime':
            return self._generate_datetime(field_spec)
        elif data_type == 'time':
            return self._generate_time(field_spec)
        elif data_type == 'email':
            return self._generate_email(field_spec)
        elif data_type == 'url':
            return self._generate_url(field_spec)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def _generate_int(self, spec: Dict) -> int:
        """Generate a random integer."""
        return self.random.randint(
            spec.get('min', 0),
            spec.get('max', 100)
        )
    
    def _generate_float(self, spec: Dict) -> float:
        """Generate a random float."""
        return round(self.random.uniform(
            spec.get('min', 0.0),
            spec.get('max', 1.0)
        ), spec.get('precision', 2))
    
    def _generate_str(self, spec: Dict) -> str:
        """Generate a random string with optional prefix and suffix."""
        length = spec.get('length', 10)
        charset = spec.get('charset', string.ascii_letters + string.digits)
        random_part = ''.join(self.random.choices(charset, k=length))
        return f"{spec.get('prefix', '')}{random_part}{spec.get('suffix', '')}"
    
    def _generate_date(self, spec: Dict) -> str:
        """Generate a random date string."""
        start = spec.get('start', datetime(2000, 1, 1))
        end = spec.get('end', datetime.now())
        random_date = start + timedelta(days=self.random.randint(0, (end - start).days))
        return random_date.strftime(spec.get('format', '%Y-%m-%d'))
    
    def _generate_list(self, spec: Dict) -> list:
        """Generate a random-length list if size range is provided."""
        element_spec = spec.get('element')
        if isinstance(spec.get('size'), list) and len(spec['size']) == 2:
            min_size, max_size = spec['size']
            size = self.random.randint(min_size, max_size)
        else:
            size = spec.get('size', 5)
        return [self._generate_field(element_spec) for _ in range(size)]
    
    def _generate_tuple(self, spec: Dict) -> tuple:
        """Generate a fixed-length tuple."""
        size = spec.get('size', 3)
        element_spec = spec.get('element')
        return tuple(self._generate_field(element_spec) for _ in range(size))
    
    def _generate_dict(self, spec: Dict) -> dict:
        """Generate a random dictionary."""
        fields = spec.get('fields', {})
        return {key: self._generate_field(field_spec) for key, field_spec in fields.items()}
    
    def _generate_choice(self, spec: Dict) -> Any:
        """Randomly select a value from the given choices."""
        choices = spec.get('choices', [])
        if not choices:
            raise ValueError("Choice type requires a 'choices' list")
        return self.random.choice(choices)
    
    def _generate_datetime(self, spec: Dict) -> Union[datetime, str]:
        """Generate a random datetime."""
        start = spec.get('start', datetime(2000, 1, 1))
        end = spec.get('end', datetime.now())
        random_seconds = self.random.randint(0, int((end - start).total_seconds()))
        random_datetime = start + timedelta(seconds=random_seconds)
        if 'format' in spec:
            return random_datetime.strftime(spec['format'])
        return random_datetime
    
    def _generate_time(self, spec: Dict) -> Union[time, str]:
        """Generate a random time."""
        start_time = spec.get('start', time(0, 0, 0))
        end_time = spec.get('end', time(23, 59, 59))
        
        # Convert times to seconds since midnight
        start_secs = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
        end_secs = end_time.hour * 3600 + end_time.minute * 60 + end_time.second
        
        # Handle cases where end time is earlier than start time (crossing midnight)
        if end_secs < start_secs:
            end_secs += 24 * 3600
        
        random_secs = self.random.randint(start_secs, end_secs) % (24 * 3600)
        hours, remainder = divmod(random_secs, 3600)
        minutes, seconds = divmod(remainder, 60)
        random_time = time(int(hours), int(minutes), int(seconds))
        
        if 'format' in spec:
            return random_time.strftime(spec['format'])
        return random_time
    
    def _generate_email(self, spec: Dict) -> str:
        """Generate a random email address."""
        username_length = spec.get('username_length', 8)
        domains = spec.get('domains', ['example.com', 'test.com', 'demo.org'])
        username = ''.join(self.random.choices(string.ascii_lowercase + string.digits, k=username_length))
        domain = self.random.choice(domains)
        return f"{username}@{domain}"
    
    def _generate_url(self, spec: Dict) -> str:
        """Generate a random URL."""
        protocols = spec.get('protocols', ['http', 'https'])
        domains = spec.get('domains', ['example.com', 'test.com', 'demo.org'])
        path_length = spec.get('path_length', 5)
        
        protocol = self.random.choice(protocols)
        domain = self.random.choice(domains)
        path = ''.join(self.random.choices(string.ascii_lowercase + string.digits, k=path_length))
        return f"{protocol}://{domain}/{path}"


# 使用示例
if __name__ == "__main__":
    # 创建数据生成器
    sampler = DataSampler(seed=42)
    
    # 定义数据结构
    schema = {
        'num_samples': 5,  # 生成5个样本
        
        # 数值型字段
        'id': {'type': 'int', 'min': 1000, 'max': 9999},
        'score': {'type': 'float', 'min': 0.0, 'max': 100.0, 'precision': 1},
        'rating': {'type': 'float', 'min': 1.0, 'max': 5.0, 'precision': 1},
        'age': {'type': 'int', 'min': 18, 'max': 65},
        
        # 其他类型字段
        'name': {'type': 'str', 'length': 8, 'prefix': 'user_'},
        'is_active': {'type': 'bool'},
        'birth_date': {'type': 'date', 'start': datetime(1980, 1, 1)},
        'status': {'type': 'choice', 'choices': ['active', 'inactive', 'pending']},
        'last_login': {'type': 'datetime', 'format': '%Y-%m-%d %H:%M:%S'},
        
        # 嵌套结构
        'address': {
            'type': 'dict',
            'fields': {
                'street': {'type': 'str', 'length': 15, 'prefix': 'Street_'},
                'city': {'type': 'str', 'length': 10},
                'zipcode': {'type': 'str', 'length': 5, 'charset': string.digits}
            }
        },
        
        # 列表类型 (可变长度)
        'phone_numbers': {
            'type': 'list',
            'size': [1, 3],  # 随机生成1-3个电话号码
            'element': {'type': 'str', 'length': 10, 'charset': string.digits}
        },
        
        # 多层嵌套
        'employment_history': {
            'type': 'list',
            'size': [0, 2],  # 随机生成0-2个就业记录
            'element': {
                'type': 'dict',
                'fields': {
                    'company': {'type': 'str', 'length': 12},
                    'position': {'type': 'str', 'length': 15, 'suffix': '_position'},
                    'years': {'type': 'int', 'min': 1, 'max': 10}
                }
            }
        }
    }
    
    # 生成样本并获取统计结果
    result = sampler.generate(**schema)
    samples = result['samples']
    statistics = result['statistics']
    
    # 打印样本
    import pprint
    
    print("\nGenerated Samples:")
    pprint.pprint(samples, width=120, depth=3)
    
    # 打印统计结果
    print("\nStatistics:")
    for field, stats in statistics.items():
        print(f"\nField: {field}")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.4f}" if isinstance(value, float) else f"  {stat_name}: {value}")