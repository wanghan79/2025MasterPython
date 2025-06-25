import random
import string
from datetime import datetime, timedelta

def generate_random_data(**kwargs):
    def _get_random_value(data_spec):
        """根据数据规范生成单个随机值"""
        data_type = data_spec['type']  # 获取数据类型
        data_range = data_spec.get('range')  

        if data_type == int:
            return random.randint(data_range[0], data_range[1])  
        elif data_type == float:
            return random.uniform(data_range[0], data_range[1])  
        elif data_type == str:
            length = data_range
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        elif data_type == bool:
            return random.choice([True, False])  
        elif data_type == list:
            length = data_spec.get('length', 1)  
            item_spec = data_range  
            return [_get_random_value(item_spec) for _ in range(length)] 
        elif data_type == tuple:
            length = data_spec.get('length', 1)  
            item_spec = data_range  
            return tuple(_get_random_value(item_spec) for _ in range(length))  
        elif data_type == dict:
            return _generate_nested_structure(data_range)  
        elif data_type == 'date':
            start_date, end_date = data_range
            time_delta = end_date - start_date
            random_days = random.randint(0, time_delta.days)
            return start_date + timedelta(days=random_days)
        else:
            return None  # 不支持的类型返回None

    def _generate_nested_structure(structure_spec):
        """递归生成嵌套结构的随机数据"""
        if isinstance(structure_spec, dict):
            # 如果是带'type'的规范，直接生成对应类型的值
            if 'type' in structure_spec:
                return _get_random_value(structure_spec)
            
            # 否则递归生成字典的每个键值对
            generated_data = {}
            for key, value_spec in structure_spec.items():
                generated_data[key] = _generate_nested_structure(value_spec)
            return generated_data
        elif isinstance(structure_spec, list):
            # 递归生成列表的每个元素
            return [_generate_nested_structure(item_spec) for item_spec in structure_spec]
        elif isinstance(structure_spec, tuple):
            # 递归生成元组的每个元素
            return tuple(_generate_nested_structure(item_spec) for item_spec in structure_spec)
        else:
            raise ValueError("不支持的类型")

    num_samples = kwargs.get('num_samples', 1)  
    structure_template = kwargs.get('structure')  

    if not structure_template:
        raise ValueError("缺少参数") 

    samples = []
    for _ in range(num_samples):
        samples.append(_generate_nested_structure(structure_template))  
    return samples

if __name__ == '__main__':
    user_data_template = {
        'user_id': {'type': int, 'range': (1000, 9999)},  
        'username': {'type': str, 'range': 10}, 
        'is_active': {'type': bool}, 
        'profile': {  
            'email': {'type': str, 'range': 15},  
            'age': {'type': int, 'range': (18, 90)}, 
            'interests': {  
                'type': list,
                'range': {'type': str, 'range': 5},  
                'length': random.randint(2, 5)  
            },
            'last_login': {'type': 'date', 'range': (datetime(2023, 1, 1), datetime.now())}  
        },
        'settings': {  
            'notifications': {'type': bool},  
            'preferences': { 
                'theme': {'type': str, 'range': 7},  
                'language': {'type': tuple, 'range': {'type': str, 'range': 2}, 'length': 2}  
            }
        },
        'transaction_history': {  
            'type': list,
            'length': 3,  
            'range': {  
                'type': dict,
                'range': {
                    'transaction_id': {'type': str, 'range': 8},  
                    'amount': {'type': float, 'range': (10.0, 500.0)},  
                    'timestamp': {'type': 'date', 'range': (datetime(2024, 1, 1), datetime.now())},  
                    'items': {  
                        'type': list,
                        'length': 2,  
                        'range': {
                            'type': dict,
                            'range': {
                                'item_id': {'type': int, 'range': (1, 100)}, 
                                'quantity': {'type': int, 'range': (1, 5)}  
                            }
                        }
                    }
                }
            }
        }
    }

    # 生成3个用户数据样本并打印
    generated_samples = generate_random_data(num_samples=3, structure=user_data_template)

    for i, sample in enumerate(generated_samples):
        print(f"--- Sample {i+1} ---")
        import json
        print(json.dumps(sample, indent=4, default=str))  

    
    simple_structure = {
        'product_id': {'type': int, 'range': (100, 200)},  
        'name': {'type': str, 'range': 5}, 
        'tags': {'type': list, 'range': {'type': str, 'range': 3}, 'length': 3}  
    }
    simple_samples = generate_random_data(num_samples=2, structure=simple_structure)
    for i, sample in enumerate(simple_samples):
        print(f"--- Simple Sample {i+1} ---")
        import json
        print(json.dumps(sample, indent=4, default=str))