import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import os.path as osp
import logging
from torch.utils.data.sampler import Sampler

@torch.no_grad()
def weight_reset(m: nn.Module):
    # - check if the current module has reset_parameters & if it's callabed called it on m
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def discrepancy(out1, out2):
    out2_t = out2.clone()
    out2_t = out2_t.detach()
    out1_t = out1.clone()
    out1_t = out1_t.detach()
    #return (F.kl_div(F.log_softmax(out1), out2_t) + F.kl_div(F.log_softmax(out2), out1_t)) / 2
    #return F.kl_div(F.log_softmax(out1), out2_t, reduction='none')
    return (F.kl_div(F.log_softmax(out1), out2_t, reduction='none')
    +F.kl_div(F.log_softmax(out2), out1_t, reduction='none')) / 2
    #return F.kl_div(F.log_softmax(out1), out2_t, reduction='batchmean')


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self,
                 num_classes,
                 epsilon=0.1,
                 use_gpu=True,
                 size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1,
            targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 -
                   self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (-targets * log_probs).mean(0).sum()
        else:
            loss = (-targets * log_probs).sum(1)
        return loss


def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent


def cal_acc_(loader, netF, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            output_f = netF.forward(inputs)  # a^t
            outputs=netC(output_f)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent


def cal_acc_proto(loader, netF, netC,proto):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netF.forward(inputs)  # a^t
            #outputs=F.normalize(outputs,dim=-1,p=2)
            #outputs = netC(output_f)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
        all_output_np=np.array(all_output)
        center=proto
        center = center.float().detach().cpu().numpy()
        dist=torch.from_numpy(cdist(all_output_np,center))
        _, predict = torch.min(dist, 1)
        accuracy = torch.sum(
            torch.squeeze(predict).float() == all_label).item() / float(
                all_label.size()[0])
    return accuracy, accuracy


def cal_acc_sda(loader, netF,netC,t=0):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs,_ = netF.forward(
                inputs,t=t)  # a^t
            outputs = netC(outputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent





def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])


def image_target(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


def image_shift(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])




def image_test(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0],
                       np.array([int(la) for la in val.split()[1:]]))
                      for val in image_list]
        else:
            images = [val.split()[0] for val in image_list]
            labels = [int(val.split()[1]) for val in image_list]
            return images, labels
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs, labels = make_dataset(image_list, labels)

        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.labels[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def office_load(args):
    train_bs = args.batch_size  
    if args.home == True:
        ss = args.dset.split('2')[0]
        tt = args.dset.split('2')[1]
        if ss == 'a':
            s = 'Art'
        elif ss == 'c':
            s = 'Clipart'
        elif ss == 'p':
            s = 'Product'
        elif ss == 'r':
            s = 'Real_World'

        if tt == 'a':
            t = 'Art'
        elif tt == 'c':
            t = 'Clipart'
        elif tt == 'p':
            t = 'Product'
        elif tt == 'r':
            t = 'Real_World'

        s_tr, s_ts = './data/office-home/{}.txt'.format(
            s), './data/office-home/{}.txt'.format(s)

        txt_src = open(s_tr).readlines()
        dsize = len(txt_src)
        tv_size = int(0.8*dsize)
        print(dsize, tv_size, dsize - tv_size)
        s_tr, s_ts = torch.utils.data.random_split(txt_src, [tv_size, dsize - tv_size])

        t_tr, t_ts = './data/office-home/{}.txt'.format(
            t), './data/office-home/{}.txt'.format(t)
        prep_dict = {}
        prep_dict['source'] = image_train()
        prep_dict['target'] = image_target()
        prep_dict['test'] = image_test()
        train_source = ImageList(s_tr,
                                 transform=prep_dict['source'])
        test_source = ImageList(s_ts,
                                transform=prep_dict['source'])
        train_target = ImageList(open(t_tr).readlines(),
                                 transform=prep_dict['target'])
        test_target = ImageList(open(t_ts).readlines(),
                                transform=prep_dict['test'])

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source,
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dset_loaders["source_te"] = DataLoader(test_source,
                                           batch_size=train_bs * 2,#2
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dset_loaders["target"] = DataLoader(train_target,
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
    dset_loaders["test"] = DataLoader(test_target,
                                      batch_size=train_bs * 3,#3
                                      shuffle=True,
                                      num_workers=args.worker,
                                      drop_last=False)
    return dset_loaders


def get_features(loader, netF):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            gt_labels = data[1]
            inputs = inputs.cuda()

            outputs = netF(inputs)
            # outputs = F.normalize(outputs)
            if len(outputs.shape) == 1:
                outputs = outputs.reshape(1, -1)
            if start_test:
                all_output = outputs.cpu().detach().numpy()
                all_label = gt_labels.detach().numpy()
                start_test = False
            else:
                try:
                    all_output = np.concatenate((all_output, outputs.cpu().detach().numpy()), 0)
                    all_label = np.concatenate((all_label, gt_labels.detach().numpy()), 0)
                except:
                    print()
    return all_output, all_label




def eval_AP_inner(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    tot_pos = np.sum(pos_flag)

    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]
    fp = np.logical_not(tp)

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)
    return ap


def VOCap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap


def cal_map_sda(features_query, gt_labels_query, features_gallary, gt_labels_gallery):
    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    scores = - cdist(features_query, features_gallary)
    for fi in range(features_query.shape[0]):
        mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery)
        try:
            mAP_ls[gt_labels_query[fi]].append(mapi)
        except:
            print()
    mAP = np.array([np.nanmean(maps) for maps in mAP_ls]).mean()
    # print('mAP - real value: {:.3f}'.format(mAP))
    return mAP

import scipy
import scipy.spatial

def fx_calc_map_label(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, train, metric)
    ord = dist.argsort(1)

    numcases = train_labels.shape[0]
    if k == 0:
        k = numcases
    if k == -1:
        ks = [50, numcases]
    else:
        ks = [k]

    def calMAP(_k):
        _res = []
        for i in range(len(test_label)):
            order = ord[i]
            p = 0.0
            r = 0.0
            for j in range(_k):
                if test_label[i] == train_labels[order[j]]:
                    r += 1
                    p += (r / (j + 1))
            if r > 0:
                _res += [p / r]
            else:
                _res += [0]
        return np.mean(_res)

    res = []
    for k in ks:
        res.append(calMAP(k))
    return res


# def kNN_DA(net, test_loader, features_gallary, gt_labels_gallery, K, sigma):
#     features_gallary = torch.from_numpy(features_gallary).t().cuda()
#     gt_labels_gallery = torch.from_numpy(gt_labels_gallery).cuda()
#     net.eval()
#     total = 0
#     testsize = test_loader.dataset.__len__()
#     C = gt_labels_gallery.max() + 1
#     top1 = 0.
#     top5 = 0.
#     with torch.no_grad():
#         retrieval_one_hot = torch.zeros(K, C).cuda()
#         iter_test = iter(test_loader)
#         for i in range(len(test_loader)):
#             data = iter_test.next()
#             inputs = data[0].cuda()
#             targets = data[1].cuda()
#             batchSize = inputs.size(0)
#             _,features,_ = net(inputs)
#
#             dist = torch.mm(features, features_gallary)
#             yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
#             candidates = gt_labels_gallery.view(1, -1).expand(batchSize, -1)
#             retrieval = torch.gather(candidates, 1, yi)
#             retrieval_one_hot.resize_(batchSize * K, C).zero_()
#             retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
#             yd_transform = yd.clone().div_(sigma).exp_()
#             probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)),
#                               1)
#             _, predictions = probs.sort(1, True)
#
#             # Find which predictions match the target
#             correct = predictions.eq(targets.data.view(-1, 1))
#
#             top1 = top1 + correct.narrow(1, 0, 1).sum().item()
#             top5 = top5 + correct.narrow(1, 0, 5).sum().item()
#
#             total += targets.size(0)
#     return top1 * 100. / total

def kNN_DA(epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0, verbose=False):
    net.eval()
    total = 0
    testsize = testloader.dataset.__len__()

    trainsize =  trainloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()

    trainLabels = torch.LongTensor(trainloader.dataset.labels).cuda()
    C = trainLabels.max() + 1


    with torch.no_grad():

        if recompute_memory:
            transform_bak = trainloader.dataset.transform
            trainloader.dataset.transform = testloader.dataset.transform
            temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=4) ## trainloader memory
            for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
                targets = targets.cuda()
                inputs = inputs.cuda()
                batchSize = inputs.size(0)
                features = net(inputs)
                trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()
            trainLabels = torch.LongTensor(temploader.dataset.labels).cuda()
            trainloader.dataset.transform = transform_bak

        lemniscate.memory = trainFeatures.t()




        top1 = 0.
        top5 = 0.
        with torch.no_grad():
            retrieval_one_hot = torch.zeros(K, C).cuda()
            for batch_idx, (inputs, targets, indexes) in enumerate(testloader):

                inputs = inputs.cuda()
                targets = targets.cuda()
                batchSize = inputs.size(0)
                features = net(inputs)
                dist = torch.mm(features, trainFeatures[:,:trainsize])

                yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
                candidates = trainLabels.view(1, -1).expand(batchSize, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot.resize_(batchSize * K, C).zero_()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(sigma).exp_()
                probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
                _, predictions = probs.sort(1, True)

                # Find which predictions match the target
                correct = predictions.eq(targets.data.view(-1, 1))

                top1 = top1 + correct.narrow(1, 0, 1).sum().item()
                top5 = top5 + correct.narrow(1, 0, 5).sum().item()

                total += targets.size(0)

                if verbose:
                    print('Test [{}/{}]\t'
                      'Top1: {:.2f}  Top5: {:.2f}'.format(
                    total, testsize, top1 * 100. / total, top5 * 100. / total))

        # logging.info(top1 * 100. / total)

    print('Test [{}/{}]\t'
          'Top1: {:.2f}  Top5: {:.2f}'.format(
        total, testsize, top1 * 100. / total, top5 * 100. / total))

    return top1 *100./ total


def recompute_memory(epoch, net, lemniscate, trainloader):

    net.eval()
    trainFeatures = lemniscate.memory.t()
    # trainLogits = lemniscate.memory_logits.t()
    batch_size = 100

    with torch.no_grad():
        transform_bak = trainloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for batch_idx, (inputs, targets, domainIDs, indexes) in enumerate(temploader):

            # targets = targets.cuda()
            inputs = inputs.cuda()
            c_batch_size = inputs.size(0)
            features = net.net(inputs)
            features = F.normalize(features)

            trainFeatures[:, batch_idx * batch_size:batch_idx * batch_size + c_batch_size] = features.data.t()
            # trainLogits[:, batch_idx * batch_size:batch_idx * batch_size + c_batch_size] = logits.data.t()
            # if batch_idx * batch_size + c_batch_size > 5000:
            #     break

        trainLabels = torch.LongTensor(temploader.dataset.labels).cuda()
        trainloader.dataset.transform = transform_bak

    lemniscate.memory = trainFeatures.t()
    # lemniscate.memory_logits = trainLogits.t()
    lemniscate.targets_memory = trainLabels
    lemniscate.memory_first = False



# Custom Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance = F.pairwise_distance(output1, output2)
        euclidean_distance = pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def pairwise_distance(x, y):
    '''
    x: n * dx
    y: m * dy
    '''

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.sqrt(torch.pow(x - y, 2).sum(2))
    return dist






class NCECriterion(nn.Module):
    """
    Eq. (12): L_{NCE}
    """
    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data
        self.eps = 1e-7

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + self.eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + self.eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """
        Draw N samples from multinomial
        :param N: number of samples
        :return: samples
        """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj

def relation_loss(relation_score, labelS):
    loss = torch.nn.MSELoss(reduction='none')(relation_score, labelS.float()).sum() / np.sqrt(labelS.size(0))
    return loss

def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_update))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)




def linear_rampup(args, current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(self.args, epoch, warm_up)

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss



class L1Loss(nn.Module):
    r"""Creates a criterion that measures the mean absolute error (MAE) between each element in
    the input :math:`x` and target :math:`y`.
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|,
    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:
    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}
    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.
    The sum operation still operates over all the elements, and divides by :math:`n`.
    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.
    Supports real-valued and complex-valued inputs.
    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(*)`, same shape as the input.
    Examples::
        >>> loss = nn.L1Loss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(L1Loss, self).__init__()
        self.reduction = reduction


    def forward(self, input, target):
        return F.l1_loss(input, target, reduction=self.reduction)

class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.scale = scale
        self.tau = 0.07

    def forward(self, pred, label_one_hot):
        pred = F.log_softmax(pred / self.tau, dim=1)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()

class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, scale=1.0):
        super(MeanAbsoluteError, self).__init__()
        self.scale = scale
        self.tau = 0.1
        return

    def forward(self, pred, label_one_hot):
        pred = F.softmax(pred / 0.01, dim=1)
        # label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        # mae = (1. - pred * label_one_hot).sum(1)
        # mae = label_one_hot - torch.sum(label_one_hot * pred, dim=1)
        mae = ((label_one_hot - pred).abs()).sum(1)
        # mae = -pred[label_one_hot>0].log()
        # mae = (1 - pred[label_one_hot > 0]).sum()
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        return self.scale * mae.mean()

class NormalizedMeanAbsoluteError(torch.nn.Module):
    def __init__(self, scale=1.0):
        super(NormalizedMeanAbsoluteError, self).__init__()
        self.scale = scale
        return

    def forward(self, pred, label_one_hot):
        pred = F.softmax(pred, dim=1)
        normalizor = 1 / (2 * (self.num_classes - 1))
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * normalizor * mae.mean()

class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.scale = scale

    def forward(self, pred, label_one_hot):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()


class NormalizedReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedReverseCrossEntropy, self).__init__()
        self.scale = scale
        self.num_classes = num_classes

    def forward(self, pred, label_one_hot):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        normalizor = 1 / 4 * (self.num_classes - 1)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * normalizor * rce.mean()


class FocalLoss(torch.nn.Module):
    '''
        https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    '''

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, scale=1.0, gamma=0, num_classes=10, alpha=None, size_average=True):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class NCEandMAE(torch.nn.Module):
    def __init__(self, alpha, beta):
        super(NCEandMAE, self).__init__()
        self.nce = NormalizedCrossEntropy(scale=alpha)
        self.mae = MeanAbsoluteError(scale=beta)

    def forward(self, pred, labels_one_hot):
        return self.nce(pred, labels_one_hot) + self.mae(pred, labels_one_hot)

class CEandMAE(torch.nn.Module):
    def __init__(self):
        super(CEandMAE, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mae = MeanAbsoluteError()

    def forward(self, args, pred, labels):
        batch_size = pred.size(0)
        labels_one_hot = torch.zeros(batch_size, args.class_num).cuda().scatter_(1, labels.view(-1, 1), 1)
        return self.ce(pred, labels) + self.mae(pred, labels_one_hot)


class MAEWithMaxPred(torch.nn.Module):
    def __init__(self, scale=1.0):
        super(MAEWithMaxPred, self).__init__()
        self.scale = scale

    def forward(self, outputs, labels):
        preds = F.softmax(outputs, 1)
        p = preds[0, labels]
        l = torch.ones_like(p).cuda()
        return torch.abs(p - l).mean()


class MeanClusteringError(nn.Module):
    """
    Mean Absolute Error
    """

    def __init__(self, tau=1):
        super(MeanClusteringError, self).__init__()
        self.tau = tau


    def forward(self, input, labels_one_hot, threshold=1):
        pred = F.softmax(input / self.tau, dim=1)
        q = labels_one_hot
        p = ((1. - q) * pred).sum(1) / pred.sum(1)
        return (p.log()).mean()


class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)

    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1 - (Yg ** self.q)) / self.q) * self.weight[indexes] - ((1 - (self.k ** self.q)) / self.q) * \
               self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1 - (Yg ** self.q)) / self.q)
        Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)


class Qloss(nn.Module):
    """
    Mean Absolute Error
    """

    def __init__(self, q=0.3):
        super(Qloss, self).__init__()
        self.q = q


    def forward(self, p, labels):
        qry_prob = F.softmax(p, dim=-1)
        qry_prob = qry_prob[torch.arange(qry_prob.shape[0]), labels]
        loss = ((1 - (qry_prob ** self.q)) / self.q)
        loss = torch.mean(loss)
        return loss


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=65):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print (P)

    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=65):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print (P)

    return y_train, actual_noise

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print (np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    # assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print (m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


import random
def split_train_test_by_category(class_num, datapath_list, train_ratio=0.7, valid_ratio=None):
    data_list_by_category = [[] for _ in range(class_num)]
    for i in range(len(datapath_list)):
        line = datapath_list[i].strip()
        label = int(line.split(' ')[1])
        data_list_by_category[label].append(line + '\n')
    train, valid, test = [], [], []
    for i in range(len(data_list_by_category)):
        data_list_by_category_i = data_list_by_category[i]
        random.shuffle(data_list_by_category_i)
        num_train = int(len(data_list_by_category_i) * train_ratio)
        if valid_ratio:
            num_valid = int(len(data_list_by_category_i) * valid_ratio)
            train.extend(data_list_by_category_i[:num_train])
            valid.extend(data_list_by_category_i[num_train: num_train+num_valid])
            test.extend(data_list_by_category_i[num_train+num_valid:])
        else:
            train.extend(data_list_by_category_i[:num_train])
            test.extend(data_list_by_category_i[num_train:])
    if valid_ratio:
        return train, valid, test
    else:
        return train, test