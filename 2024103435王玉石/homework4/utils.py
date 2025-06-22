# -*- coding: gbk -*-       # 靠靠縂BK靠
# -*- coding: utf-8 -*-     # 靠靠縐TF-8靠靠靠
import numpy as np
import os
from keras.utils import load_img, img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator

def read_img(img_path, target_size=(256,256)):
    '''读取图片'''
    img = load_img(img_path,
                   color_mode = "grayscale",
                   target_size=target_size)
    arr = img_to_array(img) / 255.0
    return arr

def draw_img(arr):
    '''绘制图片'''
    img = array_to_img(arr)
    img.show()

def normalize(arr):
    '''根据阈值将数组中的元素设置为0和1'''
    arr = np.where(arr > 0.5, 1., 0.)
    return arr

def load_data(path, target_size=(256,256)):
    '''生成数据和标签'''
    img_path = path+"Images/"
    label_path = path+"Labels/"
    imgs_list = os.listdir(img_path)
    imgs = np.empty((len(imgs_list), target_size[0], target_size[1], 1))
    labels = np.empty((len(imgs_list), target_size[0], target_size[1], 1))
    for idx, file_name in enumerate(imgs_list):
        img = read_img(img_path+file_name, target_size)
        label = read_img(label_path+file_name, target_size)
        imgs[idx] = img
        labels[idx] = label
    return imgs, labels

def adjustData(img,mask,flag_multi_class,num_class):
    print("Image shape:", img.shape)
    print("Mask shape:", mask.shape)
    if flag_multi_class:
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        if flag_multi_class:
            new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3]))
        else:
            new_mask = np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif np.max(img) > 1:
        img = img/255
        mask = mask/255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask

def trainGenerator(
    batch_size,
    train_path,
    image_folder,
    mask_folder,
    image_color_mode = "grayscale",
    mask_color_mode = "grayscale",
    image_save_prefix  = "image",
    mask_save_prefix  = "mask",
    flag_multi_class = False,
    num_class = 2,
    save_to_dir = "save_result",
    target_size = (256,256),
    seed = 1):
    '''
    可以同时生成图像和遮罩，对imagedatagen和maskdatagen使用相同的种子，以确保image和mask的转换相同
    如果要可视化生成器的结果，请设置save_to_dir
    '''
    aug_dict = dict(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        rescale=1./255,
        horizontal_flip=True,
        fill_mode='nearest')
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)
