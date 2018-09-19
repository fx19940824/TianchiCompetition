# -*- coding: utf-8 -*-

import cv2
import os
import glob
import PIL.Image as Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


clip_size = 256;
path_positive = 'D:/deeplearning/TianchiCompetition/dataset/guangdong_round1_train2_20180916/无瑕疵样本/'
path_negative = 'D:/deeplearning/TianchiCompetition/dataset/guangdong_round1_train2_20180916/瑕疵样本/'

clip_savepath_positive = 'D:/deeplearning/TianchiCompetition/dataset/guangdong_round1_train2_20180916_clip/positive/'


def clip_imgs(path_positive, clip_savepath_positive):
    path_images = glob.glob(path_positive+'*.jpg')
    index_img = int(0)
    for path_img in path_images:
        index_img = index_img+1
        #channel order: rgb
        img = cv2.imdecode(np.fromfile(path_img,dtype=np.uint8),-1)
        size_img = img.shape
        rows = height = int(size_img[0])
        cols = width = int(size_img[1])
        channel = int(size_img[2])
        
        n_rows = int(rows//clip_size + 1)
        n_cols = int(cols//clip_size + 1)
        index = 0
        for r in range(0, n_rows-1):
            up = r*clip_size
            down = [up + clip_size, rows][r == n_rows - 1]    
            for c in range(0, n_cols-1):
                index = int(n_rows*r+c)
                left = c*clip_size
                right = [left + clip_size, cols][c == n_cols - 1]
                clip = img[up:down, left:right]
                if not down-up == clip_size and right - left == clip_size:
                    continue
                clip_filename = str(index_img) + '_' + str(index) + ".jpg"
                cv2.imwrite(clip_savepath_positive+clip_filename, clip)
        break


def gen_img(path_img, folder_save):
    datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    img = load_img(path_img)  # this is a PIL image, please replace to your own file path
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    i = 0
    for batch in datagen.flow(
         x,
         batch_size=1,
         save_to_dir=folder_save,  
         save_prefix='test',
         save_format='jpg'
    ):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely



if(__name__=='__main__'):
    if not os.path.exists(path_positive):
        print('unexist input file path')
    else:
        if not os.path.exists(clip_savepath_positive):
            os.makedirs(clip_savepath_positive)
        clip_imgs(path_positive, clip_savepath_positive)
        gen_img('test_data/toclip.jpg', 'test_data/')