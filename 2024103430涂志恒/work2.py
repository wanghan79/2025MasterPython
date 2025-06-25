# coding=utf-8
# work2

import random
from datetime import datetime, timedelta

sample_template = {
    'city': {
        'education': {
            'faculty': (50, 70),
            'learners': (800, 1200),
            'staff': (20, 40),
            'budget': (410000.5, 986553.1)
        },
        'healthcare': {
            'physicians': (40, 60),
            'caregivers': (60, 80),
            'clients': (200, 300),
            'funding': (110050.5, 426553.4)
        },
        'retail': {
            'employees': (80, 150),
            'outlets': (30, 60),
            'revenue': (310000.3, 7965453.4)
        }
    }
}


def generate_nested_samples(sample_count, **schema):
    samples = []
    for idx in range(sample_count):
        sample = _construct_sample(schema, idx)
        samples.append(sample)
    return samples


def _construct_sample(template, sample_idx):
    if isinstance(template, dict):
        return {
            f"{key}{sample_idx}": _process_template(value, sample_idx)
            for key, value in template.items()
        }
    return template


def _process_template(value_template, sample_idx):
    if isinstance(value_template, dict):
        return _construct_sample(value_template, sample_idx)

    elif isinstance(value_template, (list, tuple)):
        if len(value_template) == 2:
            return _generate_random_value(value_template)
        return [_process_template(item, sample_idx) for item in value_template]

    return value_template


def _generate_random_value(range_spec):
    if all(isinstance(x, int) for x in range_spec):
        return random.randint(*range_spec)
    elif all(isinstance(x, float) for x in range_spec):
        return random.uniform(*range_spec)
    elif all(isinstance(x, str) for x in range_spec):
        return random.choice(range_spec)
    elif all(isinstance(x, bool) for x in range_spec):
        return random.choice([True, False])
    elif all(isinstance(x, datetime) for x in range_spec):
        delta = range_spec[1] - range_spec[0]
        return range_spec[0] + timedelta(seconds=random.randint(0, delta.total_seconds()))
    return range_spec


if __name__ == '__main__':
    generated_data = generate_nested_samples(5, **sample_template)
    for i, sample in enumerate(generated_data):
        print(f"样本 {i + 1}:")
        print(sample)
        print()
