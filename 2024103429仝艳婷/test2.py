# coding=utf-8
import random


# def DataGenerate(num,**kwargs):
#     result = []
#     for i in range(num):
#         element = {}
#         for key,value in kwargs.items():
#             sub_element = {}
#             if key == 'town':
#                 for sub_key,sub_value in value.items():
#                     subb_element = {}
#                     if sub_key == 'school':
#                         for subb_key,subb_value in sub_value.items():
#                             if subb_key == 'students':
#                                 subb_element['students'] = random.randint(subb_value[0],subb_value[1])
#                             if subb_key == 'teachers':
#                                 subb_element['teachers'] = random.randint(subb_value[0],subb_value[1])
#                             if subb_key == 'others':
#                                 subb_element['others'] = random.randint(subb_value[0], subb_value[1])
#                             if subb_key == 'money':
#                                 subb_element['money'] = random.uniform(subb_value[0], subb_value[1])
#                         sub_element['school'] = subb_element
#                     if sub_key == 'hospital':
#                         for subb_key,subb_value in sub_value.items():
#                             if subb_key == 'docters':
#                                 subb_element['docters'] = random.randint(subb_value[0],subb_value[1])
#                             if subb_key == 'nurses':
#                                 subb_element['nurses'] = random.randint(subb_value[0],subb_value[1])
#                             if subb_key == 'patients':
#                                 subb_element['patients'] = random.randint(subb_value[0], subb_value[1])
#                             if subb_key == 'money':
#                                 subb_element['money'] = random.uniform(subb_value[0], subb_value[1])
#                         sub_element['hospital'] = subb_element
#                     if sub_key == 'supermarket':
#                         for subb_key, subb_value in sub_value.items():
#                             if subb_key == 'sailers':
#                                 subb_element['sailers'] = random.randint(subb_value[0], subb_value[1])
#                             if subb_key == 'shop':
#                                 subb_element['shop'] = random.randint(subb_value[0], subb_value[1])
#                             if subb_key == 'money':
#                                 subb_element['money'] = random.uniform(subb_value[0], subb_value[1])
#                         sub_element['supermarket'] = subb_element
#         str_ = key + str(i)
#         element[str_] = sub_element
#         result.append(element)
#     return result

def DataGenerate(num, **kwargs):
    result = []
    for i in range(num):
        element = {}
        for key, value in kwargs.items():
            processed_value = _process_value(value) if isinstance(value, dict) else value
            element[f"{key}{i}"] = processed_value
        result.append(element)
    return result

def _process_value(data):
    if isinstance(data, dict):
        return {k: _process_value(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        if all(isinstance(x, int) for x in data):
            return random.randint(data[0], data[1])
        elif all(isinstance(x, float) for x in data):
            return random.uniform(data[0], data[1])
    return data


if __name__ == '__main__':
    data_format = {'town':
                       {'school':
                            {'teachers':(50,70),
                             'students':(800,1200),
                             'others':(20,40),
                            'money':(410000.5,986553.1)},
                        'hospital':
                            {'docters':(40,60),
                             'nurses':(60,80),
                             'patients':(200,300),
                            'money':(110050.5,426553.4)},
                        'supermarket':
                            {'sailers':(80,150),
                             'shop':(30,60),
                            'money':(310000.3,7965453.4)}
                        }
                   }
    num_data = 5
    ret = DataGenerate(num_data,**data_format)
    for i in range(num_data):
        print(ret[i])



