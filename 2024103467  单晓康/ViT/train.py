import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet, load_mnist, load_cifar10, load_cifar100
# from vit_model import vit_base_patch16_224_in21k as create_model
from vit_model import VisionTransformer
from utils import read_split_data, train_one_epoch, evaluate


def main(args, train_loader=None, val_loader=None):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    if train_loader is not None:
        tb_writer = SummaryWriter(comment=f'vit-L_{train_loader.dataset.__class__.__name__}')
    else:
        tb_writer = SummaryWriter(comment='flowers')

    if train_loader is None:
        # 自定义数据集
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.493, 0.463, 0.393], [0.063, 0.059, 0.068])]),
            "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.493, 0.463, 0.393], [0.063, 0.059, 0.068])])}

        # 实例化训练数据集
        train_dataset = MyDataSet(images_path=train_images_path,
                                  images_class=train_images_label,
                                  transform=data_transform["train"])

        # 实例化验证数据集
        val_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=data_transform["val"])

        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   pin_memory=True,
                                                   num_workers=nw,
                                                   collate_fn=train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=val_dataset.collate_fn)

    # model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    shape = train_loader.dataset.data.shape
    in_c = 1 if len(shape) == 3 else shape[-1]
    num_classes = len(train_loader.dataset.classes)
    # model = VisionTransformer(img_size=224,
    #                           patch_size=patch_size,
    #                           in_c=in_c,
    #                           embed_dim=embed_dim,
    #                           depth=12,
    #                           num_heads=8,
    #                           representation_size=None,
    #                           num_classes=num_classes, ).to(device)

    # vit-L
    model = VisionTransformer(img_size=224,
                              in_c=in_c,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes).to(device)

    # model = VisionTransformer(img_size=224,
    #                           patch_size=16,
    #                           in_c=3,
    #                           embed_dim=768,
    #                           depth=12,
    #                           num_heads=12,
    #                           representation_size=None,
    #                           num_classes=args.num_classes, ).to(device)

    if args.weights != "" and args.resume == '':
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    start_epoch = 0
    max_acc = 0
    # 断点恢复
    if args.resume != "":
        assert os.path.exists(args.resume), "weights file: '{}' not exist.".format(args.resume)
        weights_dict = torch.load(args.resume, map_location=device)
        print(model.load_state_dict(weights_dict['net']))  # 加载模型可学习参数
        optimizer.load_state_dict(weights_dict['optimizer'])  # 加载优化器参数
        start_epoch = weights_dict['epoch'] + 1  # 设置开始的epoch
        scheduler.load_state_dict(weights_dict['scheduler'])  # 恢复scheduler的state_dict
        if 'acc' in weights_dict.keys():
            max_acc = max(max_acc, weights_dict['acc'])

    for epoch in range(start_epoch, args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > max_acc:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'scheduler': scheduler.state_dict(),
                'acc': val_acc
            }
            if not os.path.isdir("./weights"):
                os.mkdir("./weights")
            torch.save(checkpoint, f"./weights/vit-L_{train_loader.dataset.__class__.__name__}_model-best.pth")
            max_acc = val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/home/a12/dataset/flower_photos")
    parser.add_argument('--model-name', default='source VIT', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符s
    parser.add_argument('--weights', type=str, default='',  # ../VIT/pre_models/vit_base_patch16_224_in21k.pth
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # 训练中断后重启,如果不需要就设置为空字符
    parser.add_argument('--resume', type=str,
                        default='',
                        help='resume training from weights')

    opt = parser.parse_args()

    data_loaders = [load_mnist(img_size=224, batch_size=opt.batch_size),
                    load_cifar10(img_size=224, batch_size=opt.batch_size),
                    load_cifar100(img_size=224, batch_size=opt.batch_size)]

    for train_loader, val_loader in data_loaders:
        main(opt, train_loader, val_loader)

    # main(opt)
