from model import UNet
from utils import *
import cv2
import numpy as np
 
#------------------------------设置参数----------------------------------------
img_channel = 1
target_size = (256, 256)
#-----------------------------------------------------------------------------
 
#--------------------------------路径------------------------------------------
data_path = "./datasets/test/"  # 测试集路径
load_path = "./save_models/unet.h5"             # 模型路径
save_path = "./save_results/predicted_image.png"    #保存结果路径
#-----------------------------------------------------------------------------
 
#-----------------------------读取测试图片--------------------------------------
test_x, test_y = load_data(data_path, target_size)
#-----------------------------------------------------------------------------
 
#------------------------------获得模型----------------------------------------
unet = UNet(img_channel)
unet.build((None, target_size[0], target_size[1], img_channel))
unet.load_weights(load_path)
#-----------------------------------------------------------------------------
 
#------------------------------测试模型----------------------------------------
model_predict = unet.predict(test_x)
 
# 保存预测结果
pred = normalize(model_predict[0])
pred = (pred * 255).astype(np.uint8)  # 转换为0-255范围的uint8类型
cv2.imwrite(save_path, pred)          # 使用OpenCV保存图片
 
# 显示图片
draw_img(test_x[0])                      # 原始图片
draw_img(test_y[0])                      # 标签图片
draw_img(normalize(model_predict[0]))    # 预测图片
#-----------------------------------------------------------------------------

