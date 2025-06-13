import torch
import os
from PIL import Image
import json
import random
import sys
sys.path.append('../')
sys.path.append('../数据集/')
from padding import padding
from get_edges import get_edge
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
from collections import Counter

print(get_edge)

transform = transforms.Compose([
    transforms.ToTensor(),
])

def resize_image_to_fixed_shorter_side(original_image, output_image_path, fixed_size=100):
    width, height = original_image.size
    if width < height:
        scale = fixed_size / width
    else:
        scale = fixed_size / height
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image

def img_process(dir, new_path, id='train'):
    img_path_png = os.path.join(new_path, dir + ".png")
    img_path_jpg = os.path.join(new_path, dir + ".jpg")
    img_path = img_path_png if os.path.exists(img_path_png) else (
        img_path_jpg if os.path.exists(img_path_jpg) else None)
    txt_path = os.path.join(new_path, dir + ".txt")
    img = Image.open(img_path).convert('RGB')
    img = img.rotate(90, expand=True)
    if img.size[0] < 850 or img.size[1] < 104:
        img = padding(img)
    if img.size[0] > 850 or img.size[1] > 104:
        w, h = img.size[0], img.size[1]
        img = img.resize((850, 104))
    return img, txt_path

def get_id(labels, train_labels, loss_labels):
    for label in labels:
        label = re.split(r'\W+', label[0].strip())
        label = [char for char in label if char]
        index = 0
        while index < len(label):
            item = label[index]
            if ',' in item and len(item) > 1:
                label.pop(index)
                item = item.split(',')
                if '' in item:
                    old = ''
                    new = ','
                    for id in range(len(item)):
                        if item[id] == old:
                            item[id] = new
                for i in item:
                    label.insert(index, i)
                    index += 1
                index -= 1
            index += 1
        id_label = []
        for item in label:
            id = dic[item.lower()]
            id_label.append(id)
        begin_id = dic['BOS']
        end_id = dic['EOS']
        train_label = [begin_id] + id_label
        loss_label = id_label + [end_id]
        train_labels.append(train_label)
        loss_labels.append(loss_label)
    return train_labels, loss_labels

path = '/home/rootroot/tmp/ZOAaoxJ5vz/handwriting_dataset'
all_img_dir = os.listdir(path)
paths = []
imgs = []
labels = []
label_paths = []
indexs = 0
imgs_path = []
test_imgs = []
test_labels = []

for n in all_img_dir:
    new_path = os.path.join(path, n)
    if 'train' in str(new_path):
        print('收集训练集')
        dirs = set(os.path.splitext(f)[0] for f in os.listdir(new_path) if not f.startswith('.') and 'voc' not in f)
        dirs = sorted(dirs, key=lambda x: int(x))
        print('train_dirs=', dirs)
        for dir in dirs:
            img, txt_path = img_process(dir, new_path)
            with open(txt_path, 'r') as f:
                label = f.readlines()
            imgs.append(img)
            labels.append(label)
    if 'test' in str(new_path) and '.' not in str(new_path):
        print('收集测试集')
        dirs = set(os.path.splitext(f)[0] for f in os.listdir(new_path) if not f.startswith('.') and 'voc' not in f)
        dirs = sorted(dirs, key=lambda x: int(x))
        print('test_dirs=', dirs)
        for dir in dirs:
            img, txt_path = img_process(dir, new_path, id='test')
            with open(txt_path, 'r') as f:
                label = f.readlines()
            test_imgs.append(img)
            test_labels.append(label)

print('labels=', labels)
print('len(labels)=', len(labels))

with open('/home/rootroot/tmp/ZOAaoxJ5vz/handwriting_dataset/voc.json', 'r', encoding='utf-8') as file:
    dic = json.load(file)

id_labels = []
train_labels = []
loss_labels = []

train_labels, loss_labels = get_id(labels, train_labels, loss_labels)

test_label = []
test_labels, _ = get_id(test_labels, test_label, loss_labels)
print('len(train_labels)=)', len(train_labels))
print('len(test_labels)=)', len(test_labels))
print('len(test_img)=', len(test_imgs))

for index, (train_item, loss_item) in enumerate(zip(train_labels, loss_labels)):
    print(f'index={index},  train_item={train_item},  loss_item={loss_item}')
print('**' * 50)
for item in test_labels:
    print('test_label=', item)

class CustomDataset(Dataset):
    def __init__(self, images, loss_labels, pred_labels, transform=None, label_length=15):
        self.images = images
        self.loss_labels = [self.pad_or_truncate(label, label_length) for label in loss_labels]
        self.pred_labels = [self.pad_or_truncate(label, label_length) for label in pred_labels]
        self.transform = transform
        self.label_length = label_length

    def pad_or_truncate(self, label, target_length):
        if len(label) < target_length:
            return label + [492] * (target_length - len(label))
        else:
            return label[:target_length]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        loss_label = self.loss_labels[idx]
        pred_label = self.pred_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(loss_label), torch.tensor(pred_label)

custom_dataset = CustomDataset(images=imgs, loss_labels=loss_labels, pred_labels=train_labels, transform=transform)
