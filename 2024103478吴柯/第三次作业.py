from work2 import generate_samples
import math

def calc(func):
    def wrapper(*args, stats=('SUM', 'AVG', 'VAR', 'RMSE'), **kwargs):
        samples = func(*args, **kwargs)
        print("Generated samples:", samples)
        def extract_numbers(obj):
            nums = []
            if isinstance(obj, dict):
                for v in obj.values():
                    nums.extend(extract_numbers(v))
            elif isinstance(obj, list):
                for item in obj:
                    nums.extend(extract_numbers(item))
            elif isinstance(obj, (int, float)):
                nums.append(obj)
            return nums

        numbers = []
        for sample in samples:
            numbers.extend(extract_numbers(sample))

        result = {}
        if not numbers:
            for stat in stats:
                result[stat] = None
            return result

        if 'SUM' in stats:
            result['SUM'] = sum(numbers)
        if 'AVG' in stats:
            result['AVG'] = sum(numbers) / len(numbers)
        if 'VAR' in stats:
            mean = sum(numbers) / len(numbers)
            result['VAR'] = sum((x - mean) ** 2 for x in numbers) / len(numbers)
        if 'RMSE' in stats:
            mean = sum(numbers) / len(numbers)
            mse = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            result['RMSE'] = math.sqrt(mse)
        return result
    return wrapper

@calc
def generate_student_samples(sample_num=1, **kwargs):
    student_structure = {
        'name': str,
        'scores': [float],
        'profile': {
            'active': bool,
            'roommate': [str],
            'teachers': str,
        }
    }
    return generate_samples(sample_num=sample_num, structure=student_structure, **kwargs)

if __name__ == "__main__":
    samples = generate_student_samples(sample_num=1, stats=('SUM', 'AVG', 'VAR', 'RMSE'))
    print(samples)

# Generated samples: [{
# 'name': 'kkWPmV', 
# 'scores': [65.25571463441455, 23.091000200820176, 49.43994282067106], 
# 'profile': {
#   'active': False, 
#   'roommate': ['cmLTMBHdEc', 'DVIEFM', 'NKKzjBUP'], 
#   'teachers': 'MQEQeCGt'
#   }
# }]
# {
#   'SUM': 137.78665765590577, 
#   'AVG': 34.44666441397644, 
#   'VAR': 622.3799429593003, 
#   'RMSE': 24.947543826182574
# }
