import random
import string

def dataSampling(**kwargs):
    def handle_int(desc):
        drange = desc.get('datarange')
        if not isinstance(drange, tuple) or len(drange) != 2:
            raise ValueError("'datarange' must be a tuple of two integers")
        return random.randint(*drange)

    def handle_float(desc):
        drange = desc.get('datarange')
        if not isinstance(drange, tuple) or len(drange) != 2:
            raise ValueError("'datarange' must be a tuple of two floats")
        return random.uniform(*drange)

    def handle_str(desc):
        charset = desc.get('datarange')
        length = desc.get('len')
        if not isinstance(charset, str) or not isinstance(length, int) or length <= 0:
            raise ValueError("Invalid string 'datarange' or 'len'")
        return ''.join(random.choices(charset, k=length))

    def handle_bool(desc):
        return random.choice([True, False])

    def handle_list(desc):
        length = desc.get('len')
        element = desc.get('element')
        if not isinstance(length, int) or length <= 0 or element is None:
            raise ValueError("Invalid list description")
        if isinstance(element, list):
            if len(element) != length:
                raise ValueError("Mismatch between list length and elements")
            return [dataSampling(**e) for e in element]
        elif isinstance(element, dict):
            return [dataSampling(**element) for _ in range(length)]
        else:
            raise ValueError("Invalid list element type")

    def handle_tuple(desc):
        length = desc.get('len')
        element = desc.get('element')
        if not isinstance(length, int) or length <= 0 or element is None:
            raise ValueError("Invalid tuple description")
        if isinstance(element, list):
            if len(element) != length:
                raise ValueError("Mismatch between tuple length and elements")
            return tuple(dataSampling(**e) for e in element)
        elif isinstance(element, dict):
            return tuple(dataSampling(**element) for _ in range(length))
        else:
            raise ValueError("Invalid tuple element type")

    def handle_dict(desc):
        entries = desc.get('entries')
        total = desc.get('len')
        if not isinstance(total, int) or total <= 0 or entries is None:
            raise ValueError("Invalid dict structure")
        result = {}
        if isinstance(entries, list):
            if len(entries) != total:
                raise ValueError("Mismatch between dict length and entries")
            for entry in entries:
                k_type, k_desc = next(iter(entry['key'].items()))
                v_type, v_desc = next(iter(entry['value'].items()))
                key = handler_map[k_type](k_desc)
                value = handler_map[v_type](v_desc)
                result[key] = value
        elif isinstance(entries, dict):
            k_type, k_desc = next(iter(entries['key'].items()))
            v_type, v_desc = next(iter(entries['value'].items()))
            used_keys = set()
            while len(result) < total:
                key = handler_map[k_type](k_desc)
                if key in used_keys:
                    continue
                value = handler_map[v_type](v_desc)
                result[key] = value
                used_keys.add(key)
        else:
            raise ValueError("Invalid dict 'entries' format")
        return result

    def handle_object(desc):
        cls = desc.get('class')
        attrs = desc.get('attrs')
        if not callable(cls) or not isinstance(attrs, dict):
            raise ValueError("Invalid object structure")
        attr_values = {}
        for name, type_desc in attrs.items():
            t_name, t_struct = next(iter(type_desc.items()))
            attr_values[name] = handler_map[t_name](t_struct)
        return cls(**attr_values)

    handler_map = {
        'int': handle_int,
        'float': handle_float,
        'str': handle_str,
        'bool': handle_bool,
        'list': handle_list,
        'tuple': handle_tuple,
        'dict': handle_dict,
        'object': handle_object
    }

    results = [handler_map[t](d) for t, d in kwargs.items()]
    return results[0] if len(results) == 1 else results
