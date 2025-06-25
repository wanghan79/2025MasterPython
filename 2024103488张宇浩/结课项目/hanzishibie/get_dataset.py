import os
import json
import shutil
import random
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bars

# --- 1. 配置参数 ---
data_root = r'D:\不会编程\CVPR\class_project\Data\data'  # 包含501个文件夹的根目录，原始数据文件路径
char_dict_path = r'D:\不会编程\CVPR\class_project\char_dict.json'# 字符映射文件
output_dir = r'D:\不会编程\CVPR\class_project\yolo_hanzi_dataset'  # 输出 YOLO 格式数据集的目录名

# 划分比例
val_size = 0.15  # 验证集比例
test_size = 0.15  # 测试集比例 (相对于原始总数)
# train_size = 1.0 - val_size - test_size

# 边界框假设 (覆盖整个图像)
# <class_index> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
bbox_annotation = "0.5 0.5 1.0 1.0" # x_center, y_center, width, height (归一化)

# 随机种子，确保结果可复现
random_seed = 42
random.seed(random_seed)

# --- 2. 加载字符映射 ---
print("加载 Char_dict.json...")
try:
    with open(char_dict_path, 'r', encoding='utf-8') as f: # 尝试UTF-8
        char_dict_raw = json.load(f)
except UnicodeDecodeError:
    try:
        with open(char_dict_path, 'r', encoding='ansi') as f: # 如果UTF-8失败，尝试ANSI (GBK/CP936等)
            char_dict_raw = json.load(f)
            print("警告: Char_dict.json 使用 ANSI 编码读取，建议转换为 UTF-8。")
    except Exception as e:
        print(f"错误: 无法读取 Char_dict.json。请确保文件存在且编码正确 (UTF-8 或 ANSI)。错误信息: {e}")
        exit()
except FileNotFoundError:
    print(f"错误: 找不到 Char_dict.json 文件，路径: {char_dict_path}")
    exit()
except json.JSONDecodeError as e:
    print(f"错误: Char_dict.json 文件格式无效。错误信息: {e}")
    exit()





print(f"从 JSON 文件加载了 {len(char_dict_raw)} 个原始映射条目。")

# --- 扫描数据文件夹，获取实际存在的类别键 ---
print(f"扫描数据目录 '{data_root}' 以确定实际存在的类别...")
actual_unicode_keys = set()
try:
    data_folders = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    print(f"找到 {len(data_folders)} 个文件夹。")#获取实际存在的类别
    for folder_name in data_folders:
        try:
            # 尝试将文件夹名 (如 "00699") 转为整数再转为字符串 (如 "699")
            key = str(int(folder_name))
            if key in char_dict_raw: # 检查这个键是否在原始 JSON 映射中存在
                actual_unicode_keys.add(key)
        except ValueError:
            # 忽略非数字命名的文件夹
            pass
except FileNotFoundError:
     print(f"错误：无法访问指定的 data_root 目录: {data_root}。")
     exit()
except Exception as e:
    print(f"扫描数据目录时发生错误: {e}")
    exit()

if not actual_unicode_keys:
    print(f"错误：在数据目录 '{data_root}' 中没有找到任何与 JSON 文件中的键匹配的文件夹。")
    exit()

print(f"根据文件夹名称，确定了 {len(actual_unicode_keys)} 个实际存在的有效类别键。")

# ---基于实际存在的键来创建映射 ---
# 对实际存在的键进行排序，以确保 class_id 分配顺序固定
sorted_actual_keys = sorted(list(actual_unicode_keys), key=lambda x: int(x)) # 按数字大小排序

unicode_to_classid = {}#
classid_to_char = {}#实际存在的类别数量 
num_classes = 0#类别的数量

print("正在创建过滤后的类别映射...")
for i, unicode_key in enumerate(sorted_actual_keys):
    if unicode_key in char_dict_raw:
        # 获取原始字符，并清理末尾的 \u0000 (空字符)
        original_char = char_dict_raw[unicode_key]
        cleaned_char = original_char.replace('\u0000', '') # 移除空字符

        # 跳过空的类别名称 (来自 key "0": "")
        if not cleaned_char:
            print(f"警告：键 '{unicode_key}' 对应的字符为空，将跳过此类别。")
            continue

        class_id = num_classes # 分配从 0 开始的连续 ID
        unicode_to_classid[unicode_key] = class_id
        classid_to_char[class_id] = cleaned_char
        num_classes += 1 # 只有成功添加了才增加类别计数
    else:
        #理论上这里不应该发生，因为 actual_unicode_keys 已经是过滤过的
        print(f"内部错误：键 '{unicode_key}' 未在 char_dict_raw 中找到，尽管它在 actual_unicode_keys 中。")

if num_classes == 0:
    print("错误：过滤后没有有效的类别。请检查文件夹名称、JSON内容和脚本逻辑。")
    exit()

print(f"最终确定使用 {num_classes} 个有效类别进行处理。")

print("有效类别映射 (部分示例):")
# 打印前几个和后几个有效类别，检查是否正确
example_keys = list(unicode_to_classid.keys())
for i in range(min(5, num_classes)):
    key = example_keys[i]
    class_id = unicode_to_classid[key]
    char = classid_to_char[class_id]
    print(f"  Class ID: {class_id}, Key: {key}, Character: '{char}'")
if num_classes > 10:
    print("  ...")
    for i in range(max(5, num_classes - 5), num_classes):
        key = example_keys[i]
        class_id = unicode_to_classid[key]
        char = classid_to_char[class_id]
        print(f"  Class ID: {class_id}, Key: {key}, Character: '{char}'")

# --- 第 3 步扫描图像文件等）应该基于过滤后的 unicode_to_classid 工作 ---
# 



# --- 3. 发现所有图像文件 ---
print("扫描图像文件...")
# all_image_paths: 存储所有找到的图像文件路径
# all_labels: 存储每个图像对应的类别ID (与all_image_paths一一对应)
all_image_paths = []
all_labels = []

# 支持常见的图像格式 (用于glob模式匹配)
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']

# --- 3.1 扫描数据目录 ---
print(f"正在扫描根目录: {data_root}")
# found_folders: 统计找到的文件夹总数
# processed_folders: 统计成功处理(有对应类别)的文件夹数
found_folders = 0
processed_folders = 0

# 显式列出data_root下的内容，帮助调试和验证
# 注意: 这里会列出所有文件和文件夹，但只有符合数字命名规则的文件夹会被处理
try:
    root_contents = os.listdir(data_root)
    print(f"在 {data_root} 中找到 {len(root_contents)} 个项目 (文件/文件夹)")
    # print(f"前10个项目: {root_contents[:10]}") # 如果需要可以取消注释看具体名称
except FileNotFoundError:
    print(f"错误：无法访问指定的 data_root 目录: {data_root}。请确保路径正确且程序有权限访问。")
    exit()
except Exception as e:
    print(f"访问目录 {data_root} 时发生未知错误: {e}")
    exit()


# 遍历根目录下的所有项目(文件/文件夹)
# 使用tqdm显示进度条，desc参数设置进度条描述文本
for item_name in tqdm(root_contents, desc="扫描项目"):
    item_path = os.path.join(data_root, item_name)

    if os.path.isdir(item_path):
        found_folders += 1
        # 文件夹名称，例如 "00699"
        unicode_index_folder_padded = item_name

        try:
            # 尝试将带前导零的文件夹名称转换为整数，再转回普通字符串（去除前导零）
            # 例如 "00699" -> 699 -> "699"
            unicode_index_int = int(unicode_index_folder_padded)
            unicode_index_key = str(unicode_index_int)

            # 使用转换后的 key (如 "699") 在字典中查找
            if unicode_index_key in unicode_to_classid:
                processed_folders += 1
                class_id = unicode_to_classid[unicode_index_key]
                image_count_in_folder = 0
                for ext in image_extensions:
                    # 使用 glob 查找当前文件夹下所有匹配扩展名的图片
                    # 注意这里使用 item_path
                    folder_images = glob(os.path.join(item_path, ext))
                    for img_path in folder_images:
                        all_image_paths.append(img_path)
                        all_labels.append(class_id)
                        image_count_in_folder += 1


            else:
                # 如果转换后的 key 仍然找不到，显示警告 (只显示前几个)
                if processed_folders < 5 and found_folders <= 20: # 控制警告数量
                     print(f"警告: 从文件夹名称 '{unicode_index_folder_padded}' 导出的键 '{unicode_index_key}' 在 Char_dict.json 中没有对应的条目，将跳过此文件夹。")

        except ValueError:
            # 如果文件夹名称不是纯数字 (例如可能是 .git, Thumbs.db 或其他非数据文件夹)，则忽略
            if found_folders <= 10: # 只显示前几个非数字文件夹警告
                print(f"信息: 跳过非预期格式的文件夹/文件: '{unicode_index_folder_padded}'")
        except Exception as e:
             print(f"处理文件夹 {item_name} 时发生错误: {e}")



print(f"总共扫描了 {found_folders} 个文件夹。")
print(f"其中 {processed_folders} 个文件夹的名称与 Char_dict.json 中的键成功匹配。")


if not all_image_paths:
    print(f"错误: 在目录 '{data_root}' 下及其与 JSON 匹配的子文件夹中，没有找到任何支持的图像文件。")
    print(f"请再次检查：")
    print(f"1. 路径 '{data_root}' 是否正确?")
    print(f"2. 匹配的文件夹 (如 {processed_folders} 个) 中是否真的包含 {', '.join(image_extensions)} 格式的图片文件?")
    print(f"3. 文件权限是否允许读取?")
    exit()

print(f"总共找到 {len(all_image_paths)} 个图像文件。")








# --- 4. 划分数据集 ---
print("划分数据集 (Train/Validation/Test)...")
# 使用sklearn的train_test_split进行分层抽样划分，确保每个集合中的类别分布均衡
# 先分出测试集，再从剩余数据中分出验证集
# 确保标签列表是整数类型，并且图像和标签列表长度一致
assert len(all_image_paths) == len(all_labels)
labels_int = [int(label) for label in all_labels] # 确保是整数

# 第一次划分: 先分出测试集
# test_size: 测试集占总数据的比例
# stratify: 按标签分层抽样，保持类别分布
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    all_image_paths,
    labels_int,
    test_size=test_size,
    random_state=random_seed,
    stratify=labels_int # 确保测试集中类别分布与总体相似
)

# 第二次划分: 从剩余数据(train_val)中分出验证集
# 计算验证集在剩余数据中的相对比例(relative_val_size)
# 例如: 总数据1000，test_size=0.15 → 测试集150，剩余850
# val_size=0.15 → 验证集应该是150/850 ≈ 0.1765
relative_val_size = val_size / (1.0 - test_size)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths,
    train_val_labels,
    test_size=relative_val_size,
    random_state=random_seed, # 使用相同的随机种子确保可复现性（如果需要拆分独立）
    stratify=train_val_labels # 确保验证集中类别分布与 (train+val) 相似
)

print(f"划分结果:")
print(f"  训练集: {len(train_paths)} 张图片")
print(f"  验证集: {len(val_paths)} 张图片")
print(f"  测试集: {len(test_paths)} 张图片")

# --- 5. 创建 YOLO 目录结构并处理文件 ---
print(f"创建 YOLO 格式数据集到 '{output_dir}'...")

# 定义数据集划分后的路径和标签
# 使用字典存储三个数据集(train/val/test)的路径和对应标签
sets = {
    'train': (train_paths, train_labels),
    'val': (val_paths, val_labels),
    'test': (test_paths, test_labels)
}

if os.path.exists(output_dir):
    print(f"警告: 输出目录 '{output_dir}' 已存在，将清空并重新创建。")
    shutil.rmtree(output_dir)

# 创建目录结构
os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)

# --- 5.1 处理文件：复制图像并创建标注文件 ---
# 对每个数据集(train/val/test)分别处理:
# 1. 复制图像文件到目标目录
# 2. 创建对应的YOLO格式标注文件(.txt)
# 3. 使用唯一文件名避免冲突(父文件夹名+原文件名)
for set_name, (paths, labels) in sets.items():
    print(f"处理 {set_name} 集...")
    image_dir = os.path.join(output_dir, 'images', set_name)
    label_dir = os.path.join(output_dir, 'labels', set_name)

    for img_path, class_id in tqdm(zip(paths, labels), total=len(paths), desc=f"  拷贝和生成标注 ({set_name})"):
        # 文件名处理:
        # 1. 获取原始文件名（带扩展名）
        base_filename = os.path.basename(img_path)
        # 2. 获取文件名（不带扩展名）
        filename_no_ext = os.path.splitext(base_filename)[0]

        # --- 生成唯一文件名 ---
        # 由于不同文件夹下的图片可能有相同的文件名（如都叫 001.jpg），
        # 需要确保目标目录中的文件名唯一。
        # 解决方案: 使用"父文件夹名_原文件名"作为新文件名
        # 例如: "00699_001.jpg" 和 "00700_001.jpg" 可以共存
        parent_folder_name = os.path.basename(os.path.dirname(img_path))
        unique_filename_no_ext = f"{parent_folder_name}_{filename_no_ext}"
        unique_base_filename = f"{unique_filename_no_ext}{os.path.splitext(base_filename)[1]}"

        # 目标图像路径
        dest_img_path = os.path.join(image_dir, unique_base_filename)
        # 目标标注文件路径
        dest_label_path = os.path.join(label_dir, f"{unique_filename_no_ext}.txt")

        # 1. 复制图像文件
        try:
             shutil.copy2(img_path, dest_img_path) # copy2 保留元数据
        except Exception as e:
            print(f"\n错误: 无法复制文件 {img_path} 到 {dest_img_path}. 错误: {e}")
            continue # 跳过这个文件

        # 2. 创建并写入标注文件 (使用假设的边界框)
        annotation_line = f"{class_id} {bbox_annotation}\n"
        try:
            with open(dest_label_path, 'w', encoding='utf-8') as f_label:
                f_label.write(annotation_line)
        except Exception as e:
            print(f"\n错误: 无法写入标注文件 {dest_label_path}. 错误: {e}")
            # 如果写入失败，最好也把对应的图片删除，避免数据不一致
            if os.path.exists(dest_img_path):
                os.remove(dest_img_path)

# --- 6. 生成 dataset.yaml 文件 ---
# YOLO训练需要此配置文件，包含:
# - 数据集路径
# - 训练/验证/测试集路径
# - 类别数量和名称列表
print("生成 dataset.yaml...")

# 获取排序后的类别名称列表
class_names = [classid_to_char[i] for i in range(num_classes)]

# 创建 YAML 内容
# 路径可以是相对路径（相对于你运行训练脚本的位置）或绝对路径
# 这里使用相对路径，假设会在 yolo_hanzi_dataset 的上一级目录运行训练
yaml_content = f"""
# YOLOv11 dataset configuration file

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: {os.path.abspath(output_dir)}  # dataset root dir (绝对路径通常更可靠)
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test # test images (optional)

# Classes
nc: {num_classes}  # number of classes
names: {json.dumps(class_names, ensure_ascii=False)} # Use json.dumps for proper list formatting and handling non-ASCII characters

""" # ensure_ascii=False 很重要，用于正确显示汉字

yaml_path = os.path.join(output_dir, 'dataset.yaml')
try:
    with open(yaml_path, 'w', encoding='utf-8') as f_yaml:
        f_yaml.write(yaml_content)
except Exception as e:
    print(f"错误: 无法写入 dataset.yaml 文件到 {yaml_path}. 错误: {e}")

print("-" * 30)
print("数据集准备完成！")
print(f"YOLO 格式的数据集已生成在: {os.path.abspath(output_dir)}")
print(f"配置文件为: {os.path.abspath(yaml_path)}")
print("-" * 30)
print("重要提示:")
print("1. 每个图像只包含一个居中的汉字，并为其生成了覆盖整个图像的边界框。")
print("2. 检查 `dataset.yaml` 文件中的路径是否正确，特别是 `path` 字段。根据你的训练环境可能需要调整。")
print("3. 检查生成的 `labels` 文件夹中的 .txt 文件内容是否符合预期格式。")
print("4. 在开始训练前，建议随机抽查几个图片及其对应的标注文件，确保它们正确对应。")
print("5. 确保 `Char_dict.json` 中的汉字编码与你的系统或训练环境兼容。YAML 文件已使用 UTF-8 编码保存。")
print("-" * 30)
