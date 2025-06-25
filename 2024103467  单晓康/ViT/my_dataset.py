from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            img = img.convert('RGB')
            suffix = self.images_path[item].split('.')[-1]
            img.save(self.images_path[item], suffix)
            img = Image.open(self.images_path[item])
            # raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def load_mnist(root='/home/a12/dataset/cls_barchmark', img_size=28, batch_size=64):
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # 将 PIL 图像或 numpy.ndarray 转换为 Tensor
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集的均值和标准差
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),  # 将 PIL 图像或 numpy.ndarray 转换为 Tensor
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集的均值和标准差
        ])}

    train_dataset = datasets.MNIST(root=root, train=True, transform=transform['train'], download=True)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform['val'], download=True)

    train_num = len(train_dataset)
    test_num = len(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print("using mnist_set {} images for training, {} images for val.".format(train_num, test_num))
    return train_loader, test_loader


def load_cifar10(root='/home/a12/dataset/cls_barchmark', img_size=32, batch_size=64):
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # 将 PIL 图像或 numpy.ndarray 转换为 Tensor
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616])  # MNIST 数据集的均值和标准差
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),  # 将 PIL 图像或 numpy.ndarray 转换为 Tensor
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616])  # MNIST 数据集的均值和标准差
        ])}

    train_dataset = datasets.CIFAR10(root=root, train=True, transform=transform['train'], download=True)
    test_dataset = datasets.CIFAR10(root=root, train=False, transform=transform['val'], download=True)

    train_num = len(train_dataset)
    test_num = len(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print("using cifar10_set {} images for training, {} images for val.".format(train_num, test_num))
    return train_loader, test_loader


def load_cifar100(root='/home/a12/dataset/cls_barchmark', img_size=32, batch_size=64):
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # 将 PIL 图像或 numpy.ndarray 转换为 Tensor
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])  # MNIST 数据集的均值和标准差
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),  # 将 PIL 图像或 numpy.ndarray 转换为 Tensor
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])  # MNIST 数据集的均值和标准差
        ])}

    train_dataset = datasets.CIFAR100(root=root, train=True, transform=transform['train'], download=True)
    test_dataset = datasets.CIFAR100(root=root, train=False, transform=transform['val'], download=True)

    train_num = len(train_dataset)
    test_num = len(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print("using cifar100_set {} images for training, {} images for val.".format(train_num, test_num))
    return train_loader, test_loader
