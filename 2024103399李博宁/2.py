
import random
import string
from datetime import date, timedelta
def get_rand_int():
    return random.randint(0, 100)
def get_rand_float():
    return round(random.uniform(0, 100), 2)

def get_rand_string(length=9):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))

def get_rand_bool():
    return random.choice([True, False])

def get_rand_date(start_yr=1995, end_yr=2022):
    start_dt = date(start_yr, 1, 1)
    end_dt = date(end_yr, 12, 31)
    delta_days = (end_dt - start_dt).days
    return (start_dt + timedelta(days=random.randint(0, delta_days))).isoformat()
TYPE_MAP = {
    'integer': get_rand_int,
    'decimal': get_rand_float,
    'text': get_rand_string,
    'boolean': get_rand_bool,
    'date_iso': get_rand_date
}
def build_item_from_schema(schema_part):
    if isinstance(schema_part, dict):
        return {k: build_item_from_schema(v) for k, v in schema_part.items()}
    elif isinstance(schema_part, list):
        return [build_item_from_schema(schema_part[0]) for _ in range(random.randint(1, 4))]
    elif isinstance(schema_part, tuple):
        return tuple(build_item_from_schema(item) for item in schema_part)
    elif isinstance(schema_part, str):
        if schema_part.startswith("text_len:"):
            length_val = int(schema_part.split(":")[1])
            return get_rand_string(length_val)
        elif schema_part in TYPE_MAP:
            return TYPE_MAP[schema_part]()
        else:
            raise ValueError(f"Unknown type spec: {schema_part}")
    else:
        raise TypeError(f"Invalid schema part type: {type(schema_part)}")
def DataProducer(num_records=1, **data_blueprint):
    if not data_blueprint:
        raise ValueError("Data blueprint must be provided.")
    if num_records <= 0:
        return []

    results = []
    for _ in range(num_records):
        results.append(build_item_from_schema(data_blueprint))
    return results

