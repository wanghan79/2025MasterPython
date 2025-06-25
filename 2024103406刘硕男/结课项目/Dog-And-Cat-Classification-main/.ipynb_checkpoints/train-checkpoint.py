#CBAMAlexNet
import csv
import os
import time
import torch
import MyNN
import torch.nn as nn
import transforms
from MyDataset import MyDataset
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('-d','--dataset',default='PetImages',type=str,help='dataset of to be trained')
parser.add_argument('--epochs',default=100,type=int,help='number of total epochs to run')
parser.add_argument('--batch_size',default=256,type=int,help='train batchsize')
parser.add_argument('--lr',default=0.001,type=float,help='learning rate')
parser.add_argument('--dropout',default=0.5,type=float,help='dropout ratio')
parser.add_argument('--momentum',default=0.9,type=float)
parser.add_argument('--model',default='lw_alexnet',type=str)
parser.add_argument('--num_classes',default=2,type=int)
parser.add_argument('--train_image_size',default=-1,type=int,help='The number of training images, -1 means all')
parser.add_argument('--test_image_size',default=-1,type=int,help='The number of test images, -1 means all')
parser.add_argument('--drop_last',default=False,type=bool)
parser.add_argument('--shuffle',default=True,type=bool)
parser.add_argument('--save_csv',default=True,type=bool)
parser.add_argument('--random_seed',default=False,type=bool)
parser.add_argument('--seed',default=42,type=int)
parser.add_argument('--learn_rate_decay',default=False,type=bool)
parser.add_argument('--decay_rate',default=0.98,type=float)
parser.add_argument('--use_l1',default=False,type=bool)
parser.add_argument('--use_l2',default=False,type=bool)
parser.add_argument('--lambda1',default=0.0001,type=float)
parser.add_argument('--lambda2',default=0.0001,type=float)
parser.add_argument('--use_data_aug',default=False,type=bool)
args = parser.parse_args()

if not args.random_seed:
    torch.manual_seed(args.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU启用")
    if not args.random_seed:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
else:
    device = torch.device("cpu")
    print("GPU用不了")
# 遍历并输出所有参数及其值
for arg, value in args.__dict__.items():
    print(f"{arg}: {value}")

if args.use_data_aug is False:
    train_transform = transforms.Compose(
        [
            transforms.Resize(224, antialias=True),
            transforms.CenterCrop(224),
            transforms.ZeroOneNormalize(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

else:
    train_transform = transforms.Compose(
        [

            transforms.Resize(224,antialias=True),
            transforms.RandomRotation(10),  # 随机旋转（-10到10度之间）
            transforms.RandomCrop(224),
            transforms.ZeroOneNormalize(),
            transforms.RandomHorizontalFlip(),  # 水平翻转
            transforms.RandomErasing(),  # 随机擦除
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
test_transform = transforms.Compose(
    [
        transforms.Resize(224, antialias=True),
        transforms.CenterCrop(224),
        transforms.ZeroOneNormalize(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

result = {'train_loss':[],'train_accuracy':[],'test_loss':[],'test_accuracy':[],"current_lr":[]}
if args.model == 'lw_alexnet':
    model = MyNN.LW_AlexNet(num_classes=args.num_classes,dropout=args.dropout).to(device)
elif args.model == 'hw_alexnet':
    model = MyNN.HW_AlexNet(num_classes=args.num_classes,dropout=args.dropout).to(device)
elif args.model == 'alexnet':
    model = MyNN.AlexNet(num_classes=args.num_classes,dropout=args.dropout).to(device)
elif args.model == 'alexnet1':
    model = MyNN.AlexNet1(num_classes=args.num_classes,dropout=args.dropout).to(device)
elif args.model == 'alexnet2':
    model = MyNN.AlexNet2(num_classes=args.num_classes,dropout=args.dropout).to(device)
elif args.model == 'alexnet_cbam_1':
    model = MyNN.AlexNet_CBAM_1(num_classes=args.num_classes,dropout=args.dropout).to(device)
elif args.model == 'alexnet_cbam_2':
    model = MyNN.AlexNet_CBAM_2(num_classes=args.num_classes,dropout=args.dropout).to(device)
elif args.model == 'alexnet_cbam_3':
    model = MyNN.AlexNet_CBAM_3(num_classes=args.num_classes,dropout=args.dropout).to(device)
elif args.model == 'alexnet_cbam_4':
    model = MyNN.AlexNet_CBAM_4(num_classes=args.num_classes,dropout=args.dropout).to(device)
else:
    raise  ValueError('Invalid model type:{}'.format(args.model))
# 交叉熵
criterion = nn.CrossEntropyLoss().to(device)
# SGD优化器
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
if args.learn_rate_decay:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.decay_rate)
data = MyDataset("data\{}".format(args.dataset), [str(n) for n in range(args.num_classes)], transform=train_transform,device=device,imgSize=args.train_image_size)
test_data = MyDataset("data\{}".format(args.dataset), [str(n) for n in range(args.num_classes)], transform=test_transform,device=device,isTrain=False, imgSize=args.test_image_size)
data_loader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0, drop_last=args.drop_last)
test_data_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0)

def train(epoch,total_epoch):
    model.train()
    total_loss = 0
    correct_predictions = 0
    with tqdm(data_loader,desc=f'第{epoch}/{total_epoch}轮训练') as t:
        for inputs,labels in t:
            labels = labels.to(device)
            # 清空梯度
            optimizer.zero_grad()
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs,labels)
            loss_item = loss.item()
            #L1 L2
            l1_regu = 0
            l2_regu = 0
            for parm in model.parameters():
                if args.use_l1:
                    l1_regu = l1_regu+torch.norm(parm,p=1)
                if args.use_l2:
                    l2_regu = l2_regu+torch.norm(parm,p=2)
            if args.use_l1 and args.use_l2:
                loss = loss+args.lambda1*l1_regu+args.lambda2*l2_regu
            elif args.use_l1:
                loss = loss+args.lambda1*l1_regu
            elif args.use_l2:
                loss = loss+args.lambda2*l2_regu



            # 反向传播
            loss.backward()
            # 更新梯度
            optimizer.step()
            # 更新显示
            t.set_postfix(loss=loss_item)
            total_loss+=loss_item
            with torch.no_grad():
                max_prob_index = torch.argmax(outputs,dim=1)
                # 累加正确预测的数量
                correct_predictions+=torch.sum(max_prob_index == labels).item()
        current_lr = optimizer.param_groups[0]['lr']
        # 更新学习率
        if args.learn_rate_decay:
            scheduler.step()
        average_loss = total_loss/len(data_loader)
        average_accuracy = correct_predictions/len(data)
        return average_loss,average_accuracy,current_lr

def test(epoch,total_epoch):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with tqdm(test_data_loader,desc=f'第{epoch}/{total_epoch}轮测试') as t:
        with torch.no_grad():
            for inputs,labels in t:
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                total_loss += loss.item()
                max_prob_index = torch.argmax(outputs,dim=1)
                correct_predictions += torch.sum(max_prob_index == labels).item()
    average_accuracy = correct_predictions/len(test_data)
    average_loss = total_loss/len(test_data_loader)
    return average_loss,average_accuracy

def main():
    best = 0
    last_file_name = ''
    for i in range(1,args.epochs+1):
        train_loss,train_accuracy,current_lr = train(i,args.epochs)
        test_loss,test_accuracy = test(i,args.epochs)
        result['train_loss'].append(train_loss)
        result['train_accuracy'].append(train_accuracy*100)
        result['test_loss'].append(test_loss)
        result['test_accuracy'].append(test_accuracy*100)
        result['current_lr'].append(current_lr)
        if best<test_accuracy:
            file_name = "{}_epoch{}_accuracy{}.pth".format(args.model,i,test_accuracy*100)
            torch.save(model,file_name)
            best = test_accuracy
            if last_file_name != '':
                os.remove(last_file_name)
            last_file_name = file_name
        print("dataset:train\t accuracy:{:.3f}%\t loss:{:.5f}\t lr:{:.10f}".format(train_accuracy*100,train_loss,current_lr))
        print("dataset:test\t accuracy:{:.3f}%\t loss:{:.5f}".format(test_accuracy*100,test_loss))
    if args.save_csv:
        with open("{}.csv".format(args.model),'w',newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Epoch','Train Loss','Train Accuracy','Test Loss','Test Accuracy','current_lr'])
            for epoch in range(len(result['train_loss'])):
                train_loss = result['train_loss'][epoch]
                train_accuracy = result['train_accuracy'][epoch]
                test_loss = result['test_loss'][epoch]
                test_accuracy = result['test_accuracy'][epoch]
                current_lr = result['current_lr'][epoch]
                csv_writer.writerow([epoch + 1, train_loss, train_accuracy, test_loss, test_accuracy,current_lr])



if __name__ == '__main__':
    main()