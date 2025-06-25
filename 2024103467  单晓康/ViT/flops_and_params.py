#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------%
# Created by "Suogh" at 2023/10/23 20:33                                 %
#                                                                       %
#       E-mail:     gaohuisuo@outlook.com                               %
#       GitHub:     https://github.com/Suogh                            %
#                                                                       %
# ----------------------------------------------------------------------%
# -- coding: utf-8 --
import torch
from vit_model import vit_base_patch16_224_in21k as create_model
from vit_model import Attention
from thop import profile

# Your Model
print('==> Building model..')
model = create_model(num_classes=5)
# model = Attention(dim=224, num_heads=8)

dummy_input = torch.randn(1, 3, 224, 224)
flops, params, layerinfo = profile(model, (dummy_input,),ret_layer_info=True)
f = open('./flops_and_params.txt', mode='w')
print(layerinfo,file=f)
f.close()
print('flops: ', flops, 'params: ', params)
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
