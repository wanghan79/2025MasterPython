import random
from pprint import pprint


def generate_sample_records(record_count, **config):
    return [
        {
            f"{region_type}{idx}": {
                facility: {
                    role: random.randint(min_val, max_val)
                    for role, (min_val, max_val) in roles.items()
                }
                for facility, roles in institutions.items()
            }
            for region_type, institutions in config.items()
        }
        for idx in range(record_count)
    ]


CONFIG_TEMPLATE = {
    'town': {
        'park': {'gardeners': (5, 15), 'visitors': (100, 500), 'maintenance_staff': (5, 10)},
        'restaurant': {'chefs': (10, 20), 'waiters': (15, 30), 'customers': (50, 200)},
        'mall': {'salespersons': (50, 100), 'shoppers': (300, 1000), 'security': (10, 20)}
    }
}

if __name__ == '__main__':
    dataset_generator = lambda: generate_sample_records(10, **CONFIG_TEMPLATE)

    for ds_num in range(1, 4):
        print(f"\n{'=' * 30}\n数据集 #{ds_num}\n{'=' * 30}")
        [pprint(record) or print("-" * 40) for record in dataset_generator()]