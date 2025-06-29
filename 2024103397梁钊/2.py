import random
import string
import numpy as np

def generate_random_sample(structure, seed=None):
    """
    根据给定的数据结构生成随机样本
    
    参数:
        structure (dict): 描述数据结构的字典
        seed (int, optional): 随机数种子，用于复现结果
        
    返回:
        any: 生成的随机样本
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
  
    if "int" in structure:
        min_val = structure.get("int", {}).get("min", 0)
        max_val = structure.get("int", {}).get("max", 100)
        return random.randint(min_val, max_val)
    
    elif "float" in structure:
        min_val = structure.get("float", {}).get("min", 0.0)
        max_val = structure.get("float", {}).get("max", 100.0)
        return random.uniform(min_val, max_val)
    
    elif "str" in structure:
        length = structure.get("str", {}).get("length", 10)
        chars = structure.get("str", {}).get("chars", string.ascii_letters + string.digits)
        return ''.join(random.choice(chars) for _ in range(length))
    
    elif "list" in structure:
        element_count = structure.get("list", {}).get("count", 5)
        element_structure = structure.get("list", {}).get("element", {"int": {}})
        return [generate_random_sample(element_structure, seed) for _ in range(element_count)]
    
    elif "dict" in structure:
        keys = structure.get("dict", {}).get("keys", ["key1", "key2"])
        value_structure = structure.get("dict", {}).get("values", {"int": {}})
        return {key: generate_random_sample(value_structure, seed) for key in keys}
    
    elif "tuple" in structure:
        element_count = structure.get("tuple", {}).get("count", 5)
        element_structure = structure.get("tuple", {}).get("element", {"int": {}})
        return tuple(generate_random_sample(element_structure, seed) for _ in range(element_count))
    
    elif "set" in structure:
        element_count = structure.get("set", {}).get("count", 5)
        element_structure = structure.get("set", {}).get("element", {"int": {}})
        # 确保集合元素唯一
        elements = []
        while len(elements) < element_count:
            new_element = generate_random_sample(element_structure, seed)
            if new_element not in elements:
                elements.append(new_element)
        return set(elements)
    
    else:
      
        return generate_random_sample({"int": {}}, seed)

def generate_random_samples(**kwargs):
    """
    生成指定数量的任意嵌套数据类型的随机样本
    
    参数:
        **kwargs: 关键字参数，格式为 {结构描述: 样本数量}
                  例如: list_int={"list": {"element": {"int": {}}}, count=5}
        
    返回:
        dict: 包含生成的样本集，键为结构描述的简化表示，值为样本列表
    """
    results = {}
    
    for struct_desc, count in kwargs.items():
     
        struct_type = next((k for k in ["list", "dict", "tuple", "set", "int", "float", "str"] 
                           if k in struct_desc), "unknown")
     
        samples = [generate_random_sample(struct_desc, seed=i) for i in range(count)]
        results[f"{struct_type}_{struct_desc.get(struct_type, {}).get('element', {}).get('type', '')}"] = samples
    
    return results

if __name__ == "__main__":
    samples = generate_random_samples(
        # 生成5个包含10个整数的列表
        list_of_ints={"list": {"count": 10, "element": {"int": {"min": 0, "max": 100}}}, "count": 5},
        # 生成3个包含2个键的字典，值为浮点数
        dict_of_floats={"dict": {"keys": ["a", "b"], "values": {"float": {"min": 0.0, "max": 1.0}}}, "count": 3},
        # 生成4个包含5个字符串的元组
        tuple_of_strs={"tuple": {"count": 5, "element": {"str": {"length": 5}}}, "count": 4},
        # 生成2个包含3个整数的集合
        set_of_ints={"set": {"count": 3, "element": {"int": {}}}, "count": 2}
    )
    # 打印生成的样本
    for name, sample_list in samples.items():
        print(f"\n{name} 样本:")
        for i, sample in enumerate(sample_list):
            print(f"样本 {i+1}: {sample}")    