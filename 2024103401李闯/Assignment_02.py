import random
import string
from datetime import datetime, time, timedelta
from typing import Any, Union, Dict, List, Tuple


class DataSampler:
    def __init__(self, seed: int = None):
        """Initialize the random data generator with an optional seed."""
        self.random = random.Random(seed)

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
        start_seconds = spec.get('start', time(0, 0, 0))
        end_seconds = spec.get('end', time(23, 59, 59))

        # Convert times to seconds since midnight
        start_secs = start_seconds.hour * 3600 + start_seconds.minute * 60 + start_seconds.second
        end_secs = end_seconds.hour * 3600 + end_seconds.minute * 60 + end_seconds.second

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
        'num_samples': 3,  # 生成3个样本

        # 基本类型
        'id': {'type': 'int', 'min': 1000, 'max': 9999},
        'name': {'type': 'str', 'length': 8, 'prefix': 'user_'},
        'score': {'type': 'float', 'min': 0.0, 'max': 100.0, 'precision': 1},
        'is_active': {'type': 'bool'},
        'birth_date': {'type': 'date', 'start': datetime(1980, 1, 1)},

        # 新增类型
        'status': {'type': 'choice', 'choices': ['active', 'inactive', 'pending']},
        'last_login': {'type': 'datetime', 'format': '%Y-%m-%d %H:%M:%S'},
        'alarm_time': {'type': 'time', 'format': '%H:%M:%S'},
        'contact_email': {'type': 'email', 'domains': ['company.com', 'org.net']},
        'profile_url': {'type': 'url', 'domains': ['mysite.com', 'webapp.io']},

        # 嵌套结构
        'address': {
            'type': 'dict',
            'fields': {
                'street': {'type': 'str', 'length': 15, 'prefix': 'Street_'},
                'city': {'type': 'str', 'length': 10},
                'zipcode': {'type': 'str', 'length': 5, 'charset': string.digits}
            }
        },

        # 增强的列表类型 (可变长度)
        'phone_numbers': {
            'type': 'list',
            'size': [1, 3],  # 随机生成1-3个电话号码
            'element': {'type': 'str', 'length': 10, 'charset': string.digits}
        },

        # 元组类型
        'coordinates': {
            'type': 'tuple',
            'size': 2,
            'element': {'type': 'float', 'min': -90.0, 'max': 90.0, 'precision': 4}
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

    # 生成样本
    samples = sampler.generate(**schema)

    # 打印结果
    import pprint

    print("Generated Samples:")
    pprint.pprint(samples, width=120, depth=4)