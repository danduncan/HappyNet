#!/usr/bin/env python

# Assembles file paths and ground truth labels for formatted data
# The output is a formatted list file, which is then used by the 
# network to import and label all images.
#
# Dataset must conform to a particular code for filenames:
#    *_HA*.tiff (or .png)
#    Where * can be anything except underscores
#   HA = two letter code for the emotion label for the image (i.e. abc_AN1.5.tiff woud be labeled 'angry')
#   Other two letter codes are in the code below
# 
# This particular naming convention is borrowed from the Jaffe open-source dataset:
# Jaffe = Japanese Female Facial Expressions (free download on their website)
# Author: Gautam Shine

# To assemble data into LMDB:
# ./listfile.py
# [caffe root]/build/tools/convert_imageset --resize_height=[h] --resize_width=[w] [dataset root] [image paths/labels file] [lmdb name]
# Example:
# /home/gshine/Documents/caffe/build/tools/convert_imageset --resize_height=224 --resize_width=224 datasets/jaffe/ jaffe_list.txt jaffe_train_lmdb
# /Users/Dan/Development/caffe/build/tools/convert_imageset --resize_height=224 --resize_width=224 datasets/training_images/ datasets/training_list.txt datasets/training_set_lmdb

# To train net:
# [caffe root] train -solve [solver file] -weights [model file]
# Example:
# /home/gshine/Documents/caffe/build/tools/caffe train -solver models/VGG_S_rgb/solver.prototxt -weights models/VGG_S_rgb/EmotiW_VGG_S.caffemodel
# /Users/Dan/Development/caffe/build/tools/caffe train -solver models/Custom_Model/solver.prototxt -weights models/Custom_Model/EmotiW_VGG_S.caffemodel

import os, glob

categoriesEitW = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']

jaffe_categories_map = {
    'HA': categoriesEitW.index('Happy'),
    'SA': categoriesEitW.index('Sad'),
    'NE': categoriesEitW.index('Neutral'),
    'AN': categoriesEitW.index('Angry'),
    'FE': categoriesEitW.index('Fear'),
    'DI': categoriesEitW.index('Disgust'),
    'SU': categoriesEitW.index('Surprise')
    }

def get_label(fname):
    label = fname.split('.')[1][0:2]
    return jaffe_categories_map[label]

# File and label list to input to caffe
f = open('datasets/training_list.txt', 'w')

# List of images to train on
# Include png for homemade images, and tiff for jaffe images
dir = 'datasets/training_images/'
imgList = glob.glob(dir+'*.png') + glob.glob(dir+'*.tiff')


for img in imgList:
    if os.path.isdir(img):
        continue
    label = get_label(img)
    fname = img.split('/')[2]
    f.write(fname + ' ' + str(label) + '\n')

f.close()
