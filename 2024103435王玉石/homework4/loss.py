#coding=gbk
import keras.backend as K


def dice_coef(y_true, y_pred):
    '''dice系数'''
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    '''dice损失函数'''
    return 1. - dice_coef(y_true, y_pred)

