import random
import string
from datetime import datetime, timedelta


class DataSampler:
    def __init__(self, random_generator=None):
        self.random_generator = random_generator or random.SystemRandom()



    def generate_value(self, value_def):
        if "type" not in value_def:
            return value_def

        data_type = value_def["type"]
        params = {k: v for k, v in value_def.items() if k != "type"}

        if data_type == "int":
            datarange = params.get("datarange", (0, 70))
            return self.random_generator.randint(datarange[0], datarange[1])

        elif data_type == "float":
            datarange = params.get("datarange", (0.0, 70.0))
            return self.random_generator.uniform(datarange[0], datarange[1])

        elif data_type == "str":
            length = params.get("len", 13)
            datarange = params.get("datarange", string.ascii_letters + string.digits)
            return ''.join(self.random_generator.choice(datarange) for _ in range(length))

        elif data_type == "bool":
            return self.random_generator.choice([True, False])

        elif data_type == "date":
            start_str, end_str = params.get("datarange", ("2001-01-01", "2030-01-01"))
            start = datetime.strptime(start_str, "%Y-%m-%d")
            end = datetime.strptime(end_str, "%Y-%m-%d")
            delta = end - start
            random_days = self.random_generator.randint(0, delta.days)
            return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")

        elif data_type == "list":
            list_length = params.get("len", 5)
            element_def = params["elements"]
            return [self.generate_value(element_def) for _ in range(list_length)]

        elif data_type == "tuple":
            tuple_length = params.get("len", 5)
            element_def = params["elements"]
            return tuple(self.generate_value(element_def) for _ in range(tuple_length))

        elif data_type == "dict":
            return {k: self.generate_value(v) for k, v in params["subs"].items()}

        else:
            return None

    def generate(self, **kwargs):
        sample_dict = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                sample_dict[key] = self.generate_value(value)
            else:
                sample_dict[key] = value
        return sample_dict


    def generate_data(self):

        order_structure = {
            "order_id": {
                "type": "str",
                "datarange": string.ascii_uppercase + string.digits,
                "len": 15
            },
            "user_id": {
                "type": "int",
                "datarange": (10000, 99999)
            },
            "order_date": {
                "type": "date",
                "datarange": ("2001-01-01", "2030-01-01")
            },
            "total_amount": {
                "type": "float",
                "datarange": (10.0, 1000.0)
            },
            "is_paid": {
                "type": "bool"
            },
            "payment_method": {
                "type": "str",
                "datarange": ["Credit Card", "PayPal", "Alipay", "WeChat Pay", "Apple Pay"],
                "len": 1
            },
            "shipping_info": {
                "type": "dict",
                "subs": {
                    "address": {
                        "type": "str",
                        "datarange": string.ascii_letters + string.digits + " ",
                        "len": 27
                    },
                    "city": {
                        "type": "str",
                        "datarange": ["Beijing", "Shanghai", "New York", "London", "Berlin", "Tokyo", "Seoul"],
                        "len": 1
                    },
                    "zip_code": {
                        "type": "str",
                        "datarange": string.digits,
                        "len": 7
                    }
                }
            },
            "items": {
                "type": "list",
                "len": self.random_generator.randint(1, 8),
                "elements": {
                    "type": "dict",
                    "subs": {
                        "product_id": {
                            "type": "int",
                            "datarange": (10000, 99999)
                        },
                        "quantity": {
                            "type": "int",
                            "datarange": (1, 10)
                        },
                        "price": {
                            "type": "float",
                            "datarange": (10.0, 999.99)
                        }
                    }
                }
            },
            "coupon_code": {
                "type": "str",
                "datarange": string.ascii_uppercase + string.digits,
                "len": 8
            },
            "status_history": {
                "type": "list",
                "len": self.random_generator.randint(2, 6),
                "elements": {
                    "type": "str",
                    "datarange": ["Pending", "Processing", "Awaiting Payment", "Shipped", "Delivered", "Cancelled", "Returned"],
                    "len": 1
                }
            },
            "shipment_progress": {
                "type": "tuple",
                "len": self.random_generator.randint(2, 5),
                "elements": {
                    "type": "date",
                    "datarange": ("2001-01-01", "2025-06-30")
                }
            }
        }

        return self.generate(**order_structure)




if __name__ == "__main__":
    sampler = DataSampler()
    sample_one = sampler.generate_data()
    print(sample_one)
