###############################################################################
# Caffe VGG_S net emotion classification
#
# This file contains utility functions for interactions with the Caffe
# deep learning framework.
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

from utility_functions import *

# Load mean caffe image
def loadMeanCaffeImage(img="mean_training_image.binaryproto",curDir="datasets/"):
  mean_filename=os.path.join(curDir,img)
  proto_data = open(mean_filename, "rb").read()
  a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
  mean  = caffe.io.blobproto_to_array(a)[0]
  return mean

# Display an image (input is numpy array)
def showimage(img):
    if img.ndim == 3:
        img = img[:, :, ::-1]
    plt.set_cmap('jet')
    plt.imshow(img,vmin=0, vmax=0.2)

# Display network activations
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # Force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # Tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    showimage(data)

# Plot the last image and conv1 layer's weights and responses
def plot_layer(input_image, VGG_S_Net, layer):
    plt.figure(1)
    _ = plt.imshow(input_image)

    plt.figure(2)
    filters = VGG_S_Net.params[layer][0].data
    vis_square(filters.transpose(0, 2, 3, 1))

    plt.figure(3)
    feat = VGG_S_Net.blobs[layer].data[0]
    vis_square(feat)

    plt.show(block=False)

# RGB dimension swap + resize
# Depending on how an image was imported, sometimes it will be XxYxRGB and 
# other times it will be RGBxXxY.
# This function takes either as input, and it always returns RGBxXxY.
def mod_dim(img, x=256, y=256, c=3):
    # Resize only if necessary:
    if not np.array_equal(img.shape,[c,x,y]):
        resized = caffe.io.resize_image(img, (x,y,c)) # (256, 256, 3)
        rearranged = np.swapaxes(np.swapaxes(resized, 1, 2), 0, 1) # (3,256,256)
    else:
        rearranged = img

    return rearranged

# Calculate mean image over list of image filenames
# Can also return the mean image if it is already saved as "mean.binaryproto"
def compute_mean(input_list, plot_mean=False):
    # If no data supplied, use mean supplied with pretrained model
    if len(input_list) == 0:
        net_root = '.'
        net_dir = 'VGG_S_rgb'
        mean_filename=os.path.join(net_root, net_dir, 'mean.binaryproto')
        proto_data = open(mean_filename, "rb").read()
        a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
        mean  = caffe.io.blobproto_to_array(a)[0]
    else:
        x,y,c = 256,256,3
        mean = np.zeros((c, x, y))
        for img_file in input_list:
            img = caffe.io.load_image(img_file)
            img = mod_dim(img, x, y, c)
            mean += img
        mean /= len(input_list)

        # Plot the mean image if desired:
        if plot_mean:
            plt.imshow(np.swapaxes(np.swapaxes(mean, 0, 1), 1, 2))
            plt.show()
    return mean

# Given filename for mean image, and the directory containing a network file
# Construct and return a Caffe network object
# Note: For legacy reasons, this assumes your model is stored in:
# ./models/[net_dir]/EmotiW_VGG_S.caffemodel
#  where net_dir is supplied by the user
def make_net(mean=None, net_dir='VGG_S_rgb'):
    # net_dir specifies a root directory containing a *.caffemodel file
    # Options in our setup are: VGG_S_[rgb / lbp / cyclic_lbp / cyclic_lbp_5 / cyclic_lbp_10]

    # This should hopefully already be in your system path, but just to be sure:
    caffe_root = '/home/Users/Dan/Development/caffe/'
    sys.path.insert(0, caffe_root + 'python')

    # Configure matplotlib
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # Generate paths to the various model files
    net_root = 'models'
    net_pretrained = os.path.join(net_root, net_dir, 'EmotiW_VGG_S.caffemodel')
    net_model_file = os.path.join(net_root, net_dir, 'deploy.prototxt')

    # Construct Caffe network object
    VGG_S_Net = caffe.Classifier(net_model_file, net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
    return VGG_S_Net

# Load a minibatch of images
# Inputs:   List of image filenames, 
#           Color boolean (true if images are in color), 
#           List of labels corresponding to each image, 
#           Index of first image to load
#           Number of images to load
# Output:   List of image numpy arrays of size Num x (W x H x 3)
#           List of labels for just the images in the batch
def load_minibatch(input_list, color, labels, start,num):
    # Enforce minimum on start
    start = max(0,start)

    # Enforce maximum on end
    end = start + num
    end = min(len(input_list), end)

    # Isolate files
    files = input_list[start:end]

    images = []
    for file in files:
        img = caffe.io.load_image(file, color)
        
        # Handle incorrect image dims for uncropped images
        # TODO: Get uncropped images to import correctly
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
        
        # BUG FIX: Is this ok?
        # color=True gets the correct desired dimension of WxHx3
        # But color=False gets images of WxHx1. Need WxHx3 or will get "Index out of bounds" exception
        # Fix by concatenating three copies of the image
        if img.shape[2] == 1:
            img = cv.merge([img,img,img])

        # Add image array to batch
        images.append(img)

    labelsReduced = labels[start:end]
    return images, labelsReduced

# Big function:
# Classify all images in a list of image file names
# Using the inputs, constructs a network, imports images either individually or in minibatches,
# gets the network classification, and builds up the confusion matrix.
# No return value, but it can plot the confusion matrix at the end
def classify_emotions(input_list, color, categories, labels, plot_neurons, plot_confusion,useMean=True):
    # Compute mean
    #mean = compute_mean(input_list)
    if useMean:
      mean = loadMeanCaffeImage()
    else:
      mean = None


    # Create VGG_S net with mean
    VGG_S_Net = make_net(mean,net_dir='Custom_Model')
    
    # Classify images in directory
    conf_mat = [] # tuples to be passed to confusion matrix generator

    numImages = len(input_list)

    # Due to network architecture, using minibatches does not speed anything up
    # (at least for datasets of up to 3000 images)
    miniBatch = False
    if miniBatch:
        i = 0
        batchSize = 500

        metrics = [] # Will hold tuples of timing metrics for all batches

        totalLoad, totalPredict = 0, 0

        while i < numImages:

            t = time.time()
            images,labelsReduced = load_minibatch(input_list, color, labels, i, batchSize)
            loadTime = time.time() - t
            totalLoad += loadTime
            print 'Batch of  ' + str(len(images)) + '  images.'

            # images is a list of input images
            # Input images should be WxHx3, e.g. 490x640x3
            t = time.time()
            prediction = VGG_S_Net.predict(images, oversample=False)
            predictTime = time.time() - t
            totalPredict += predictTime

            for j in range(len(prediction)):
                pred = prediction[j]
                lab = labelsReduced[j]

                # Append (label, prediction) tuple to confusion matrix list
                conf_mat.append((lab, pred.argmax()))

                # Print results as Filename: Prediction
                #print(input_list[i+j].split('/')[-1]+': '+categories[prediction.argmax()])

            metrics.append((len(images),loadTime,predictTime))
            i += batchSize
        
        # Print all timing metrics
        print "\nTiming data for classify_emotions() (minibatch mode):"
        for i in range(len(metrics)):
            bs, ltime, ptime = metrics[i]
            print "Batch " + str(i) + " (" + str(bs) + " images):\tLoad: " + str(ltime) + "s\t Predict: " + str(ptime) + "s"
        print "\nTotal images: " + str(len(input_list))
        print "Total time loading: " + str(totalLoad) + "\t(" + str(float(totalLoad)/len(input_list)) + "s / image)"
        print "Total time predicting: " + str(totalPredict) + "\t(" + str(float(totalPredict)/len(input_list)) + "s / image)"
        print " "

    else:
        loadTime, predictTime = 0, 0

        for i in range(numImages):
            img_file = input_list[i]
            label = labels[i]

            print('File name: ', img_file)
            t = time.time()
            input_image = caffe.io.load_image(img_file)
            loadTime += time.time() - t

            # Handle incorrect image dims for uncropped images
            # TODO: Get uncropped images to import correctly
            if input_image.shape[0] == 3:
                input_image = np.swapaxes(np.swapaxes(input_image, 0, 1), 1, 2)

            # Input image should be WxHxK, e.g. 490x640x3
            t = time.time()
            prediction = VGG_S_Net.predict([input_image], oversample=False)
            predictTime += time.time() - t

            # Append (label, prediction) tuple to confusion matrix list
            conf_mat.append((label, prediction.argmax()))

            # Print results as Filename: Prediction
            print(img_file.split('/')[-1]+': '+categories[prediction.argmax()])

        # Print timing metrics:
        print "\nTiming data for classify_emotions() (serial mode):"
        print "Load time:   " + str(loadTime)    + "s\t(" + str(loadTime/numImages)    + "s / image)"
        print "Predict time:" + str(predictTime) + "s\t(" + str(predictTime/numImages) + "s / image)"
        print " "

    if plot_neurons:
        layer = 'conv1'
        plot_layer(input_image, VGG_S_Net, layer)

    # Generates confusion matrix and calculates accuracy
    confusion_matrix(conf_mat, categories, plot_confusion)


# Classify all faces in a single video frame
# Return a labels list of integer labels
def classify_video_frame(frame, faces, VGG_S_Net, categories=None):
    # Convert to float format
    # Video frames normally imported as uint32
    frame = frame.astype(np.float32)
    frame /= 255.0

    labels = []

    for x,y,w,h in faces:
        img = frame[y:y+h,x:x+w,:]

        # Input image should be WxHxK, e.g. 490x640x3
        prediction = VGG_S_Net.predict([img], oversample=False)


        labels.append(prediction.argmax())

    return labels

