# coding=gbk

from model import UNet
from loss import dice_coef, dice_coef_loss
from utils import *


#------------------------------设置参数----------------------------------------
img_heigh = 256
img_width = 256
img_channel = 1
batch_size = 8
epochs = 5
steps_per_epoch=100
#-----------------------------------------------------------------------------


#--------------------------------路径------------------------------------------
data_path = "datasets/train/"
save_path = "save_models/unet.h5"
#-----------------------------------------------------------------------------


#------------------------------加载数据----------------------------------------
myGene = trainGenerator(
    batch_size, 
    data_path, 
    image_folder='Images', 
    mask_folder='Labels', 
    target_size=(img_heigh, img_width))
#-----------------------------------------------------------------------------


#------------------------------构建模型----------------------------------------
unet = UNet(img_channel)
unet.build((None, img_heigh, img_width, img_channel))
unet.compile(
    optimizer='adam',
    loss=dice_coef_loss,
    metrics=[dice_coef]
    )
unet.summary()
#----------------------------------------------------------------------------


#-----------------------------训练保存模型-------------------------------------
unet.fit(myGene, steps_per_epoch=steps_per_epoch, epochs=epochs)
unet.save_weights(save_path)
#----------------------------------------------------------------------------
