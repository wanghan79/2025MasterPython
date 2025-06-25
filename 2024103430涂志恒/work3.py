# coding=utf-8
# work3

import math
import random
from functools import wraps
from pprint import pformat

community_structure = {
    'district': {
        'education_center': {
            'educators': (50, 70),
            'learners': (800, 1200),
            'support_staff': (20, 40),
            'budget': (410000.5, 986553.1)
        },
        'medical_center': {
            'physicians': (40, 60),
            'nursing_staff': (60, 80),
            'service_users': (200, 300),
            'funding': (110050.5, 426553.4)
        },
        'shopping_center': {
            'sales_staff': (80, 150),
            'retail_units': (30, 60),
            'turnover': (310000.3, 7965453.4)
        }
    }
}


def compute_statistics(*requested_stats):
    def decorator(data_generator_func):
        @wraps(data_generator_func)
        def wrapped_generator(*args, **kwargs):
            generated_samples = data_generator_func(*args, **kwargs)

            stat_functions = {
                'total': lambda data: sum(data),
                'average': lambda data: sum(data) / len(data),
                'variation': lambda data: sum((x - (sum(data) / len(data))) ** 2 for x in data) / len(data),
                'rmse': lambda data: math.sqrt(sum((x - (sum(data) / len(data))) ** 2 for x in data) / len(data))
            }

            if not requested_stats:
                metrics_to_compute = stat_functions
            else:
                metrics_to_compute = {
                    metric: stat_functions[metric]
                    for metric in requested_stats
                    if metric in stat_functions
                }

            numeric_values = {}
            for sample in generated_samples:
                for location_data in sample.values():
                    for path, value in _extract_numeric_values(location_data):
                        if isinstance(value, (int, float)):
                            numeric_values.setdefault(path, []).append(value)

            analysis_results = {}
            for field_path, values in numeric_values.items():
                analysis_results[field_path] = {
                    stat_name: calculator(values)
                    for stat_name, calculator in metrics_to_compute.items()
                }

            return {
                'generated_data': generated_samples,
                'statistical_analysis': analysis_results
            }

        return wrapped_generator

    return decorator


def _extract_numeric_values(nested_dict, parent_key='', separator='.'):
    items = []
    for key, val in nested_dict.items():
        current_path = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(val, dict):
            items.extend(_extract_numeric_values(val, current_path, separator))
        elif isinstance(val, (int, float)):
            items.append((current_path, val))
    return items


def create_random_samples(sample_count, **structure_template):
    samples = []
    for sample_num in range(sample_count):
        sample = {}
        for category, sub_structure in structure_template.items():
            sample[f"{category}{sample_num}"] = _generate_random_structure(sub_structure)
        samples.append(sample)
    return samples


def _generate_random_structure(template):
    if isinstance(template, dict):
        return {
            key: _generate_random_structure(value)
            for key, value in template.items()
        }
    elif isinstance(template, (list, tuple)) and len(template) == 2:
        if all(isinstance(x, int) for x in template):
            return random.randint(*template)
        elif all(isinstance(x, float) for x in template):
            return random.uniform(*template)
    return template


@compute_statistics('total', 'average', 'variation', 'rmse')
def generate_community_data(num_samples, **structure):
    return create_random_samples(num_samples, **structure)


analysis_report = generate_community_data(5, **community_structure)

print(pformat(analysis_report['statistical_analysis'], width=120, indent=2))
