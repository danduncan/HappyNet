#!/usr/bin/env python

###############################################################################
#
# This script reads in a single image, detects all faces with opencv,
# classifies emotions on each face with a neural network, adds emojis to the
# original image corresponding to each emotion, and saves the new image to file.
#
#
# Date modified: March 2016
#
# Authors:    Dan Duncan    
#             Gautam Shine
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
categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
useCNN = True
defaultEmoji = 2 # Index of default emoji (0-6)

# List your dataset root directories here:
dirImage = 'datasets/generic_image_directory'

### START SCRIPT ###

# Set up face detection
faceCascades = load_cascades()

# Set up network
if useCNN:
    mean = loadMeanCaffeImage()
    VGG_S_Net = make_net(mean,net_dir="Custom_Model")

# Get all emojis
emojis = loadAllEmojis()

# Find all image files
extensions = [".png",".jpg",".jpeg",".tiff"]
filenames = []
for ext in extensions:
  filenames += glob.glob(dirImage + "/*" + ext)

print "Processing " + str(len(filenames)) + " images"

for filename in filenames:
    print "Now processing: " + filename

    # Note: Caffe and openCV use different input formats!
    # Both images will be WxHxC shaped
    # But Caffe's units are floats from 0.0 - 1.0
    # OpenCV uses uint8 data format with values from 0 - 255
    
    # Load image in caffe format
    frame = caffe.io.load_image(filename)

    # Load same image in openCV format
    pilImg = Image.open(filename)
    cvImg = cv.cvtColor(np.array(pilImg), cv.COLOR_RGB2BGR)

    # Find all faces
    with nostdout():
        _, faces = DetectFace(cvImg,True,faceCascades,single_face=False,second_pass=False,draw_rects=False,scale=1.0)

    frame = cvImg
    if len(faces) == 0 or faces is None:
      # No faces found
      pass
    else:
        if useCNN:
            # Get a label for each face
            labels = classify_video_frame(frame, faces, VGG_S_Net, categories=None)

            # Add an emoji for each label
            frame = addMultipleEmojis(frame,faces,emojis,labels)

        else:
            # Just use the smiley face (no CNN classification)
            frame = addEmoji(frame,faces,emojis[defaultEmoji])

    # Save to file
    fn = filename.split('/')[-1]
    fn = fn.split('.')
    fn = fn[0] + '_emojis.' + fn[1]
    _ = saveTestImage(frame,outDir=dirImage,filename=fn)
    print "Image: " + fn + " saved. " + str(len(faces)) + " faces found!"


