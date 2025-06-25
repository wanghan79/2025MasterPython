import json

json_path = "/root/fssd/SMFFDDG/AAP/parameters_list.json"

with open(json_path, "r") as f:
    data = json.load(f)

print(f"总共的 key 数量为: {len(data)}")
print("所有的 key 为:", list(data.keys()))
