#!/usr/bin/env python

###############################################################################
#
# This is a quick script to convert all of the files in the open-source 
# Cohn-Kanade Plus (CK+) emotions dataset to use the same naming convention 
# as the Japanese Female Facial Expressions (JAFFE) dataset
#
# This then allows the two datasets to be merged
#
# Note that CK+ includes transitional images (i.e. a face halfway between neutral
# and strong emotion). Only neutral faces and strong emotions are included. No 
# transitional images are included.
#
# Date modified: March 2016
#
# Authors:  Dan Duncan 	 	
# 			Gautam Shine
#
###############################################################################

import os, shutil, sys, time, re, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import Image
import caffe

from caffe_functions import *
from opencv_functions import *
from utility_functions import *

### USER-SPECIFIED VARIABLES: ###

# List your dataset root directories here:
dirCKPlus = 'datasets/CK_Plus'

# Select which dataset to use (case insensitive):
dataset = 'ckplus'

# Flags:
cropFlag = True # False disables image cropping

### START SCRIPT: ###

# Set up inputs
dir = dirCKPlus
color = False
single_face = True

# Clean up and discard anything from the last run
dirCrop = dir + '/cropped'
rmdir(dirCrop)

# Master list of categories for EmotitW network
categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
suffixes   = ['AN',      'DI',       'FE',    'HA',      'NE',        'SA',   'SU']

# Load dataset image list
input_list, labels = importDataset(dir, dataset, categories)

# Perform detection and cropping if desired (and it should be desired)
mkdir(dirCrop)
input_list = faceCrop(dirCrop, input_list, color, single_face)

# Print outs
# print input_list
# print labels

# Rename all files to Jaffe format
for i in range(len(input_list)):
  # Get file info
  filename = input_list[i]
  lab = labels[i]
  labText = suffixes[lab]

  # Generate new filename
  fn = filename.split('.')
  out = fn[0] + '.' + labText + '.' + fn[1]

  # Rename file
  os.rename(filename,out)


  
