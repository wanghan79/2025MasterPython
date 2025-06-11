import random
import string
from datetime import datetime, timedelta

class DataSampler
    def __init__(self, rng=None)
        
        Initializes the data sampler.

        Args
            rng Optional random number generator, defaults to random module.
        
        self.rng = rng or random

    def random_value(self, data_type, data_params=None)
        
        Generates a random value based on the specified data type and parameters.

        Args
            data_type Data type (int, float, str, bool, list, tuple, dict, date, etc.).
            data_params Data range or other parameters.

        Returns
            The generated random value.
        
        if data_type is int
            return self.rng.randint(data_params[0], data_params[1])

        elif data_type is float
            return self.rng.uniform(data_params[0], data_params[1])

        elif data_type is str
            if isinstance(data_params, int)
                return ''.join(self.rng.choices(string.ascii_letters, k=data_params))
            elif isinstance(data_params, list)
                return self.rng.choice(data_params)

        elif data_type is bool
            return self.rng.choice([True, False])

        elif data_type is list
            return [self.random_value(data_params['type'], data_params.get('params'))
                    for _ in range(data_params['length'])]

        elif data_type is tuple
            return tuple(self.random_value(data_params['type'], data_params.get('params'))
                         for _ in range(data_params['length']))

        elif data_type is dict
            return self.generate_structure(data_params)

        elif data_type == 'date'
            start, end = data_params
            days = self.rng.randint(0, (end - start).days)
            return start + timedelta(days=days)

        else
            return None

    def generate_structure(self, data_structure)
        
        Generates a complete data sample based on the provided data structure.

        Args
            data_structure Data structure definition.

        Returns
            The generated data structure.
        
        if isinstance(data_structure, dict)
            result = {}
            for key, val in data_structure.items()
                if isinstance(val, dict)
                    data_type = val.get('type')
                    data_params = val.get('params')
                    subs = val.get('subs', [])

                    if isinstance(subs, list) and subs
                        result[key] = [self.generate_structure(sub) for sub in subs]
                    else
                        result[key] = self.random_value(data_type, data_params)
                else
                    result[key] = val
            return result
        else
            raise ValueError(Unsupported structure type)

    def generate_samples(self, data_structure, num_samples=1)
        
        Generates multiple data samples.

        Args
            data_structure Data structure definition.
            num_samples Number of samples to generate.

        Returns
            A list of generated samples.
        
        return [self.generate_structure(data_structure) for _ in range(num_samples)]


# Example usage
if __name__ == __main__
    # Define data structure
    data_structure = {
        'user_id' {'type' int, 'params' (1000, 9999)},
        'username' {'type' str, 'params' 8},
        'is_active' {'type' bool},
        'score' {'type' float, 'params' (0.0, 100.0)},
        'tags' {
            'type' list,
            'params' {
                'type' str,
                'params' 5,
                'length' 3
            }
        },
        'login_history' {
            'type' dict,
            'subs' [
                {'last_login' {'type' 'date', 'params' [datetime(2023, 1, 1), datetime(2023, 12, 31)]}},
                {'login_count' {'type' int, 'params' (0, 100)}}
            ]
        }
    }

    # Create sampler and generate samples
    sampler = DataSampler()
    samples = sampler.generate_samples(data_structure, num_samples=3)

    # Print generated samples
    for i, sample in enumerate(samples, 1)
        print(f样本 {i})
        print(sample)
        print()