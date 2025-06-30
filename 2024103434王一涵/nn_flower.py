import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms,models,datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK']='True'
data_dir = './102flowers'
train_dir = data_dir+ '/train'
valid_dir = data_dir + '/val'
data_transforms ={
    'train' : transforms.Compose([transforms.RandomRotation(45),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(p=0.025),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ]),
    'val' : transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
}
batch_size = 8
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}
dataloaders = {x: torch.utils.data. DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True) for x in ['train','val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
class_names = image_datasets['train'].classes
with open('./102flowers/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
def im_convert(tensor):
    """数据展示"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    # 下面将图像还原，使用squeeze，将函数标识的向量转换为1维度的向量，便于绘图
    # transpose是调换位置，之前是换成了（c， h， w），需要重新还原为（h， w， c）
    image = image.transpose(1, 2, 0)
    # 反正则化（反标准化）
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

    # 将图像中小于0 的都换成0，大于的都变成1
    image = image.clip(0, 1)

    return image
# 使用上面定义好的类进行画图
fig = plt.figure(figsize = (20, 12))
columns = 4
rows = 2

# iter迭代器
# 随便找一个Batch数据进行展示
dataiter = iter(dataloaders['val'])
inputs, classes = dataiter.__next__()
print(classes)
for i in range(0,8):
    print(cat_to_name[class_names[classes[i]][1:]],end=",")
for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks = [], yticks = [])
    # 利用json文件将其对应花的类型打印在图片中
    #ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
    ax.set_title(cat_to_name[class_names[classes[idx]][1:]])
    plt.imshow(im_convert(inputs[idx]))
plt.show()
model_name = 'resnet' # 可选的模型比较多['resnet', 'alexnet', 'vgg', 'squeezenet', 'densent', 'inception']
# 主要的图像识别用resnet来做
feature_extract = True
# 是否用GPU进行训练
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.   Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# 将一些层定义为false，使其不自动更新
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
# 打印模型架构告知是怎么一步一步去完成的
# 主要是为我们提取特征的

model_ft = models.resnet152()


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择适合的模型，不同的模型初始化参数不同
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """
        Resnet152
        """

        # 1. 加载与训练网络
        model_ft = models.resnet152(pretrained=use_pretrained)
        # 2. 是否将提取特征的模块冻住，只训练FC层
        set_parameter_requires_grad(model_ft, feature_extract)
        # 3. 获得全连接层输入特征
        num_frts = model_ft.fc.in_features
        # 4. 重新加载全连接层，设置输出102
        model_ft.fc = nn.Sequential(nn.Linear(num_frts, 102),
                                    nn.LogSoftmax(dim=1))  # 默认dim = 0（对列运算），我们将其改为对行运算，且元素和为1
        input_size = 224

    elif model_name == "alexnet":
        """
        Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        # 将最后一个特征输出替换 序号为【6】的分类器
        num_frts = model_ft.classifier[6].in_features  # 获得FC层输入
        model_ft.classifier[6] = nn.Linear(num_frts, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """
        VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_frts = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_frts, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """
        Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """
        Densenet
        """
        model_ft = models.desenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_frts = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_frts, num_classes)
        input_size = 224

    elif model_name == "inception":
        """
        Inception V3
        """
        model_ft = models.inception_V(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        num_frts = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_frts, num_classes)

        num_frts = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_frts, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
# 设置模型名字、输出分类数
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained = True)

# GPU 计算
model_ft = model_ft.to(device)

# 模型保存, checkpoints 保存是已经训练好的模型，以后使用可以直接读取
filename = 'checkpoint.pth'

# 是否训练所有层
params_to_update = model_ft.parameters()
# 打印出需要训练的层
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad ==True:
            print("\t", name)
for name, param in model_ft.named_parameters():
    print(name,param)
# 优化器设置
optimizer_ft  = optim.Adam(params_to_update, lr = 1e-2)
# 学习率衰减策略
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# 学习率每7个epoch衰减为原来的1/10
# 最后一层使用LogSoftmax(), 故不能使用nn.CrossEntropyLoss()来计算

criterion = nn.NLLLoss()


# 定义训练函数
# is_inception：要不要用其他的网络
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, is_inception=False, filename=filename):
    since = time.time()
    # 保存最好的准确率
    best_acc = 0
    """
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    """
    # 指定用GPU还是CPU
    model.to(device)
    # 下面是为展示做的
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]
    # 最好的一次存下来
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                # 下面是将inputs,labels传到GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    # if这面不需要计算，可忽略
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:  # resnet执行的是这里
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # 概率最大的返回preds
                    _, preds = torch.max(outputs, 1)

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 打印操作
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 模型保存
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    # tate_dict变量存放训练过程中需要学习的权重和偏执系数
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 保存训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs
#若太慢，把epoch调低，迭代50次可能好些
#训练时，损失是否下降，准确是否有上升；验证与训练差距大吗？若差距大，就是过拟合
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=5, is_inception=(model_name=="inception"))
# 将全部网络解锁进行训练
for param in model_ft.parameters():
    param.requires_grad = True

# 再继续训练所有的参数，学习率调小一点\
optimizer = optim.Adam(params_to_update, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 损失函数
criterion = nn.NLLLoss()
# 加载保存的参数
# 并在原有的模型基础上继续训练
# 下面保存的是刚刚训练效果较好的路径
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=2, is_inception=(model_name=="inception"))
#加载已经训练的模型
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU 模式
model_ft = model_ft.to(device) # 扔到GPU中

# 保存文件的名字
filename='checkpoint.pth'

# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])


# 测试数据预处理
def process_image(image_path):
    # 读取测试集数据
    img = Image.open(image_path)
    # Resize, thumbnail方法只能进行比例缩小，所以进行判断
    # 与Resize不同
    # resize()方法中的size参数直接规定了修改后的大小，而thumbnail()方法按比例缩小
    # 而且对象调用方法会直接改变其大小，返回None
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    # crop操作， 将图像再次裁剪为 224 * 224
    left_margin = (img.width - 224) / 2  # 取中间的部分
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224  # 加上图片的长度224，得到全部长度
    top_margin = bottom_margin + 224

    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    # 相同预处理的方法
    # 归一化
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # 注意颜色通道和位置
    img = img.transpose((2, 0, 1))

    return img


def imshow(image, ax=None, title=None):
    """展示数据"""
    if ax is None:
        fig, ax = plt.subplots()

    # 颜色通道进行还原
    image = np.array(image).transpose((1, 2, 0))

    # 预处理还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)

    return ax


image_path = r'./102flowers/val/c79/image_01988.jpg'
img = process_image(image_path)  # 我们可以通过多次使用该函数对图片完成处理
imshow(img)
# 得到一个batch的测试数据
dataiter = iter(dataloaders['val'])
images, labels = dataiter.__next__()

model_ft.eval()

if train_on_gpu:
    # 前向传播跑一次会得到output
    output = model_ft(images.cuda())
else:
    output = model_ft(images)

# batch 中有8 个数据，每个数据分为102个结果值， 每个结果是当前的一个概率值
output.shape
#计算得到最大概率
_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())# 将秩为1的数组转为 1 维张量
print(preds_tensor)
for i in range(0,8):
    print(cat_to_name[class_names[preds[i]][1:]],end=",")
#展示预测结果
fig = plt.figure(figsize = (20, 20))
columns = 4
rows = 2

for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks =[], yticks =[])
    plt.imshow(im_convert(images[idx]))
    #ax.set_title("{} ({})".format(cat_to_name[class_names[preds[idx]][1:]], cat_to_name[class_names[labels[idx].item()][1:]]),
                #color = ("green" if cat_to_name[class_names[preds[idx]][1:]]== cat_to_name[class_names[labels[idx].item()][1:]] else "red"))
    ax.set_title("{} ({})".format(class_names[preds[idx]], class_names[labels[idx].item()]),
                color = ("green" if class_names[preds[idx]]== class_names[labels[idx].item()] else "red"))
plt.show()
# 绿色的表示预测是对的，红色表示预测错了