# coding=utf-8
import random


def generate_sample_records(record_count, **config):
    sample_data = []
    for idx in range(record_count):
        region_record = {}
        for region_type, institutions in config.items():
            facility_data = {}
            for facility_category, staff_types in institutions.items():
                staff_numbers = {}
                for staff_role, count_range in staff_types.items():
                    staff_numbers[staff_role] = random.randint(count_range[0], count_range[1])
                facility_data[facility_category] = staff_numbers
            region_key = f"{region_type}{idx}"
            region_record[region_key] = facility_data
        sample_data.append(region_record)
    return sample_data


CONFIG_TEMPLATE = {
    'town': {
        'park': {
            'gardeners': (5, 15),
            'visitors': (100, 500),
            'maintenance_staff': (5, 10)
        },
        'restaurant': {
            'chefs': (10, 20),
            'waiters': (15, 30),
            'customers': (50, 200)
        },
        'mall': {
            'salespersons': (50, 100),
            'shoppers': (300, 1000),
            'security': (10, 20)
        }
    }
}

# 完整调用代码
if __name__ == '__main__':
    from pprint import pprint

    dataset_count = 3  # 生成3个独立数据集
    records_per_dataset = 10  # 每个数据集包含5条记录

    for ds_num in range(1, dataset_count + 1):
        print(f"\n{'=' * 30}")
        print(f" 数据集 #{ds_num} ".center(30, '='))
        print(f"{'=' * 30}\n")

        dataset = generate_sample_records(
            record_count=records_per_dataset,
            **CONFIG_TEMPLATE
        )

        for idx, record in enumerate(dataset, 1):
            print(f"第{idx}条记录：")
            pprint(record)
            print("-" * 40)