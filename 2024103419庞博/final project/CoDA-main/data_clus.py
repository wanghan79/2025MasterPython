import torch
import torch.utils.data as data
import os
import clustering
import time
from transforms import *
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from utils.utils import *

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
            return images, np.array(labels)
    return images

class ImageList_idx(Dataset):
    def __init__(self,
                 args,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB',
                 domain_id=0):
        nb_classes = args.class_num
        self.imgs, self.labels = make_dataset(image_list, labels)
        self.domain_id = domain_id


        # self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index],  self.labels[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, target, self.domain_id, index


    def __len__(self):
        return len(self.imgs)


def office_load_idx(args):
    train_bs = args.batch_size

    if args.dataset == 'office_home':
        assert args.class_num == 65
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
        else:
            raise NotImplementedError

        if tt == 'a':
            t = 'Art'
        elif tt == 'c':
            t = 'Clipart'
        elif tt == 'p':
            t = 'Product'
        elif tt == 'r':
            t = 'Real_World'
        else:
            raise NotImplementedError

        
        s_tr_path = './data_split/office-home/'+s+'_train.txt'
        s_ts_path = './data_split/office-home/' + s + '_test.txt'
        t_tr_path = './data_split/office-home/' + t + '_train.txt'
        t_ts_path = './data_split/office-home/' + t + '_test.txt'

        if os.path.exists(s_tr_path) and os.path.exists(s_ts_path):
            s_tr, s_ts = open(s_tr_path).readlines(), open(s_ts_path).readlines()
        # else:
        #     s_tr, s_ts = './data_split/office-home/{}.txt'.format(
        #         s), './data_split/office-home/{}.txt'.format(s)
        #     txt_src = open(s_tr).readlines()
        #     s_tr, s_ts = split_train_test_by_category(args.class_num,
        #                                             txt_src,
        #                                             train_ratio=0.8)
        #     with open('./data_split/office-home/'+s+'_train.txt', 'w') as f:
        #         for i in s_tr:
        #             f.write(i)
        #     with open('./data_split/office-home/'+s+'_test.txt', 'w') as f:
        #         for i in s_ts:
        #             f.write(i)
        if os.path.exists(t_tr_path) and os.path.exists(t_ts_path):
            t_tr, t_ts = open(t_tr_path).readlines(), open(t_ts_path).readlines()
        # else:
        #     t_tr, t_ts = './data_split/office-home/{}.txt'.format(
        #         t), './data/office-home/{}.txt'.format(t)
        #     txt_tgt = open(t_tr).readlines()
            
        #     t_tr, t_ts = split_train_test_by_category(args.class_num,
        #                                             txt_tgt,
        #                                             train_ratio=0.8)
        #     with open('./data_split/office-home/'+t+'_train.txt', 'w') as f:
        #         for i in t_tr:
        #             f.write(i)
        #     with open('./data_split/office-home/'+t+'_test.txt', 'w') as f:
        #         for i in t_ts:
        #             f.write(i)



    prep_dict = {}
    prep_dict['source'] = image_train()
    prep_dict['target'] = image_target()
    prep_dict['test'] = image_test()
    train_source = ImageList_idx(args, s_tr, transform=prep_dict['source'], domain_id=0)
    test_source = ImageList_idx(args, s_ts, transform=prep_dict['test'], domain_id=0)
    eval_train_source = ImageList_idx(args, s_tr, transform=prep_dict['source'], domain_id=0)
    train_target = ImageList_idx(args, t_tr, transform=prep_dict['target'], domain_id=1)
    test_target = ImageList_idx(args, t_ts, transform=prep_dict['test'], domain_id=1)
    eval_train_target = ImageList_idx(args, t_tr, transform=prep_dict['target'], domain_id=1)

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source,
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False) ## SimCLR needs drop last batch to match the batch_size
    dset_loaders["source_te"] = DataLoader(
        test_source,
        batch_size=train_bs * 3,  #2
        shuffle=False,
        num_workers=args.worker,
        drop_last=False)
    dset_loaders["source_eval_tr"] = DataLoader(eval_train_source,
                                           batch_size=train_bs,
                                           shuffle=False,
                                           num_workers=args.worker,
                                           drop_last=False)
    '''dset_loaders["source_f"] = DataLoader(fish_source,
                                           batch_size=train_bs ,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)'''
    dset_loaders["target"] = DataLoader(train_target,
                                        batch_size=train_bs,
                                        # collate_fn=collate_fn,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False) ## SimCLR needs drop last batch to match the batch_size
    dset_loaders["test"] = DataLoader(
        test_target,
        batch_size=train_bs * 3,  #3
        shuffle=False,
        num_workers=args.worker,
        drop_last=False)
    dset_loaders["target_eval_tr"] = DataLoader(eval_train_target,
                                        batch_size=train_bs,
                                        # collate_fn=collate_fn,
                                        shuffle=False,
                                        num_workers=args.worker,
                                        drop_last=False)
    return dset_loaders




def compute_features(args, dataloader, model, N):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()

    # discard the label information in the dataloader
    for i, (input_tensor, _, noisy_labels, targets, domain_id, indexes) in enumerate(dataloader):
        input_var = input_tensor.cuda()
        aux = model.net(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')
            domainIDs = []
            indexes_list = []

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch_size: (i + 1) * args.batch_size] = aux

        else:
            # special treatment for final batch
            features[i * args.batch_size:] = aux

        domainIDs.extend(domain_id.tolist())
        indexes_list.extend(indexes.tolist())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features, domainIDs, indexes_list
