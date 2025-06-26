import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
from typing import List, Optional, Tuple
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}


class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x



vgg_dict = {"vgg16_bn":models.vgg16_bn}


class VggBase(nn.Module):
    def __init__(self, vgg_name):
        super(VggBase, self).__init__()
        model_vgg = vgg_dict[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.avgpool = model_vgg.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0:4]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


    # Instead of manually picking channels, deploying embedding layer to generate random domain attention. Can be fixed or also updatd during adaptation.
class feat_bootleneck_sdaE(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck_sdaE, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type
        self.em = nn.Embedding(2, 256)

    def forward(self, x,t,s=100,all_mask=False):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        out=x
        if t == 0:
            t = torch.LongTensor([t]).cuda()
            self.mask = nn.Sigmoid()(self.em(t) * s)
            flg = torch.isnan(self.mask).sum()
            out = out * self.mask
        if t == 1:
            t_ = torch.LongTensor([0]).cuda()
            self.mask = nn.Sigmoid()(self.em(t_) * s)
            t = torch.LongTensor([t]).cuda()
            mask = nn.Sigmoid()(self.em(t) * s)
            out = out * mask
        if all_mask:
            t0 = torch.LongTensor([0]).cuda()
            t1 = torch.LongTensor([1]).cuda()
            mask0 = nn.Sigmoid()(self.em(t0) * s)
            mask1 = nn.Sigmoid()(self.em(t1) * s)
            self.mask=mask0
            out0 = out * mask0
            out1 = out * mask1
        if all_mask:
            return (out0,out1), (self.mask,mask1)
        else:
            return out, self.mask


# manually generating domain attention
class feat_bootleneck_sda(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck_sda, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x, t, s=100, all_mask=False):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        out = x
        if t == 0:
            mask_s = torch.zeros(256).cuda()
            mask_s[range(int(0.75 * 256))] = 1
            out = out * mask_s
        if t == 1:
            mask_s = torch.zeros(256).cuda()
            mask_s[range(int(0.75 * 256))] = 1
            mask_t = torch.zeros(256).cuda()
            mask_t[range(int(0.25 * 256), 256)] = 1
            out = out * mask_t
        if all_mask:
            mask_s = torch.zeros(256).cuda()
            mask_s[range(int(0.75 * 256))] = 1
            out0 = out * mask_s
            mask_t = torch.zeros(256).cuda()
            mask_t[range(int(0.25 * 256), 256)] = 1
            out1 = out * mask_t
        if all_mask:
            return (out0, out1), (mask_s, mask_t)
        else:
            return out, mask_s


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x



# Instead of manually picking channels, deploying embedding layer to generate random domain attention. Can be fixed or also updatd during adaptation.
class ResNet_sdaE(nn.Module):
    def __init__(self):
        super().__init__()
        model_resnet = torchvision.models.resnet50(True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        #self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu,
                                            self.maxpool, self.layer1,
                                            self.layer2, self.layer3,
                                            self.layer4, self.avgpool)
        #self.bottle=nn.Sequential(nn.Linear(2048, 256),nn.BatchNorm1d(256))
        self.bottle = nn.Linear(2048, 256)
        self.bn = nn.BatchNorm1d(256)
        self.em = nn.Embedding(2, 256)
        self.mask = torch.empty(1, 256)

    def forward(self, x, t, s=100, all_out=False):
        out = self.feature_layers(x)
        out = out.view(out.size(0), -1)
        out = self.bottle(out)
        out = self.bn(out)
        #t=0
        if t == 0:
            t = torch.LongTensor([t]).cuda()
            self.mask = nn.Sigmoid()(self.em(t) * s)
            flg = torch.isnan(self.mask).sum()
            if flg != 0:
                print('nan occurs')
            #print(self.mask.shape)
            out = out * self.mask
        if t == 1:
            t_ = torch.LongTensor([0]).cuda()
            self.mask = nn.Sigmoid()(self.em(t_) * s)
            t = torch.LongTensor([t]).cuda()
            mask = nn.Sigmoid()(self.em(t) * s)
            flg = torch.isnan(mask).sum()
            if flg != 0:
                print('nan occurs')
            #print(self.mask.shape)
            out = out * mask
        if all_out:
            t0 = torch.LongTensor([0]).cuda()
            t1 = torch.LongTensor([1]).cuda()
            mask0 = nn.Sigmoid()(self.em(t0) * s)
            mask1 = nn.Sigmoid()(self.em(t1) * s)
            self.mask = mask0
            out0 = out * mask0
            out1 = out * mask1

        if all_out:
            return (out0, out1), (self.mask, mask1)
        else:
            return out, self.mask






class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.
    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer).
    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])
    """

    def __init__(
        self,
        blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ):
        super(ProjectionHead, self).__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Computes one forward pass through the projection head.
        Args:
            x:
                Input of shape bsz x num_ftrs.
        """
        return self.layers(x)



class NNCLRProjectionHead(ProjectionHead):
    """Projection head used for NNCLR.
    "The architectureof the projection MLP is 3 fully connected layers of sizes
    [2048,2048,d] where d is the embedding size used to apply the loss. We use
    d = 256 in the experiments unless otherwise stated. All fully-connected
    layers are followed by batch-normalization [36]. All the batch-norm layers
    except the last layer are followed by ReLU activation." [0]
    [0]: NNCLR, 2021, https://arxiv.org/abs/2104.14548
    """
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 2048,
                 output_dim: int = 256):
        super(NNCLRProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, nn.BatchNorm1d(output_dim), None),
        ])

class NNCLRPredictionHead(ProjectionHead):
    """Prediction head used for NNCLR.
    "The architecture of the prediction MLP g is 2 fully-connected layers
    of size [4096,d]. The hidden layer of the prediction MLP is followed by
    batch-norm and ReLU. The last layer has no batch-norm or activation." [0]
    [0]: NNCLR, 2021, https://arxiv.org/abs/2104.14548
    """
    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 4096,
                 output_dim: int = 256):
        super(NNCLRPredictionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class NNCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = NNCLRProjectionHead(512, 512, 128)
        self.prediction_head = NNCLRPredictionHead(128, 512, 128)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return y, z, p

class RelationDNN(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """

    def __init__(self, embed_size, hidden_dim, output_dim):
        super(RelationDNN, self).__init__()
        self.Sequential = nn.Sequential(nn.Linear(embed_size, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, output_dim)
                                        )

    def forward(self, src_emb, tgt_emd):
        # sim_all = []
        # n_src = src_emb.size(0)
        # n_tgt = tgt_emd.size(0)
        # for i in range(n_tgt):
        #     tgt_emd_i = tgt_emd[i].unsqueeze(0)
        #     tgt_emd_i_expand = tgt_emd_i.repeat(n_src, 1)
        #     sim_vec = torch.cat((src_emb, tgt_emd_i_expand), 1)
        #     sim_i = self.Sequential(sim_vec)
        #     sim_i = F.sigmoid(sim_i)
        #     sim_all.append(sim_i)
        # sim_all = torch.cat(sim_all, 1)
        ni = src_emb.size(0)
        di = src_emb.size(1)
        nt = tgt_emd.size(0)
        dt = tgt_emd.size(1)
        src_emb = src_emb.unsqueeze(1).expand(ni, nt, di)
        src_emb = src_emb.reshape(-1, di)

        tgt_emd = tgt_emd.unsqueeze(0).expand(ni, nt, dt)
        tgt_emd = tgt_emd.reshape(-1, dt)

        y = torch.cat((src_emb, tgt_emd), 1)
        # y = y_I * y_T

        relation_score = self.Sequential(y)
        relation_score = F.sigmoid(relation_score)
        return relation_score

import resnet
from LinearAverage import LinearAverage

class Model(nn.Module):
    def __init__(self, args, memorySize_s, memorySize_t):
        super(Model, self).__init__()
        self.args = args
        if args.arch == 'resnet50':
            self.net = resnet.resnet50(pretrained=args.pretretrained, low_dim=args.low_dim)
        elif args.arch == 'resnet18':
            self.net = resnet.resnet18(pretrained=args.pretretrained, low_dim=args.low_dim)
        else:
            raise NotImplementedError

        self.lemniscate_s = LinearAverage(args.low_dim, memorySize_s, args.nce_t, args.nce_m)
        self.lemniscate_t = LinearAverage(args.low_dim, memorySize_t, args.nce_t, args.nce_m)
        ndata = memorySize_s + memorySize_t
        # self.lemniscate = LinearAverage(args.class_num, args.low_dim, ndata, args.n_neighbor, args.nce_t, args.nce_m)

        self.top_layer = nn.Linear(args.low_dim, args.class_num)

        self.optimizer_tl = torch.optim.SGD(self.top_layer.parameters(),
                                            lr=1e-4,
                                            momentum=0.9,
                                            weight_decay=0)

        self.top_layer_src = nn.ModuleList()
        self.top_layer_tgt = nn.ModuleList()

        if args.cluster_num_list:
            for i in range(len(args.cluster_num_list)):
                cluster_num = args.cluster_num_list[i]
                self.top_layer_src.append(nn.Linear(args.low_dim, cluster_num, bias=False))
                self.top_layer_tgt.append(nn.Linear(args.low_dim, cluster_num, bias=False))

            self.optimizer_tl_src = torch.optim.SGD(
                [elem for i in range(len(args.cluster_num_list)) for elem in list(self.top_layer_src[i].parameters())],
                lr=1e-4, 
                momentum=0.9,
                weight_decay=0)
            self.optimizer_tl_tgt = torch.optim.SGD(
                [elem for i in range(len(args.cluster_num_list)) for elem in list(self.top_layer_tgt[i].parameters())],
                lr=1e-4,
                momentum=0.9,
                weight_decay=0)

        # self.top_layer.apply(init_weights)
        self.optimizer = torch.optim.SGD(self.net.parameters(),
                                         lr=args.lr,
                                         momentum=0.9,
                                         weight_decay=1e-5)
        self.projector = nn.Sequential(
            nn.Linear(args.low_dim, args.low_dim, bias=False),
            nn.ReLU(),
            nn.Linear(args.low_dim, 64, bias=False),
        )
        self.optimizer_proj = torch.optim.SGD(self.projector.parameters(),
                                         lr=args.lr,
                                         momentum=0.9,
                                         weight_decay=1e-5)

        # self.optimizer_tl = torch.optim.SGD([elem for i in range(len(args.cluster_nums)) for elem in list(self.top_layer[i].parameters())],
        #                                         lr=1e-4,
        #                                         momentum=0.9,
        #                                         weight_decay=0)

        # self.optimizer = torch.optim.Adam(self.net.parameters(),
        #                                  lr=0.0001)
        # self.optimizer_tl = torch.optim.SGD(self.top_layer.parameters(),
        #                                     lr=0.0001)

        self.scheduler = StepLR(self.optimizer, step_size=7, gamma=0.1)
        # self.scheduler_tl = StepLR(self.optimizer_tl, step_size=7, gamma=0.1)

    def train_start(self):
        """switch to train mode"""
        self.net.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.net.eval()

    def forward_emb(self, images):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.cuda()
        # Forward feature encoding
        img_embs = self.net(images)
        return img_embs

    def forward_simclr(self, images):
        h = self.net(images)
        z = self.projector(h)
        return h, z



    def forward(self, x, head=None):
        emd = self.net(x)
        # x = F.relu(x)
        if head == 'src':
            x = [self.top_layer_src[i](emd) for i in range(len(self.args.cluster_num_list))]
        elif head == 'tgt':
            x = [self.top_layer_tgt[i](emd) for i in range(len(self.args.cluster_num_list))]
        elif not head:
            x = self.top_layer(emd)
        return x, emd


