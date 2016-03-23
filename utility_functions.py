###############################################################################
# Utility functions for OpenCV-Caffe chaining
# and anything generally useful for our other scripts
#
# Date modified: March 2016
#
# Authors:    Dan Duncan    
#             Gautam Shine
#
###############################################################################

import os, shutil, sys, time, re, glob, csv
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import Image
import caffe
import datetime
from PIL import Image
from opencv_functions import *
import contextlib, cStringIO

# Plot confusion matrix
def plot_confusion_matrix(cm, names=None, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(4)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # Add labels to confusion matrix:
    if names is None:
        names = range(cm.shape[0])

    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)

    plt.tight_layout()
    plt.ylabel('Correct label')
    plt.xlabel('Predicted label')
    plt.show()

# Generate confusion matrix for Jaffe
# results = list of tuples of (correct label, predicted label)
#           e.g. [ ('HA', 3) ]
# categories = list of category names
# Returns confusion matrix; rows are correct labels and columns are predictions
def confusion_matrix(results, categories, plotConfusion=False):
    # Empty confusion matrix
    matrix = np.zeros((7,7))

    # Iterate over all labels and populate matrix
    for label, pred in results:
        matrix[label,pred] += 1
        #matrix[map_categories[label],pred] += 1

    # Print matrix and percent accuracy
    accuracy = float(np.trace(matrix))/len(results)
    print('Confusion Matrix: ')
    print(matrix)
    print 'Accuracy: ' +  str(accuracy*100) + '%'

    # Normalize confusion matrix
    normalizeMatrix = True
    if normalizeMatrix:
      print "utility.confusion_matrix(). Non-normalized conf_mat:"  
      print matrix
      s = np.sum(matrix,1) # Sum each row
      for i in range(matrix.shape[0]):
        # Normalization handles class imbalance in training set
        matrix[i,:] /= s[i]

    # Save matrix to file:
    np.save("confusion_matrix.npy",matrix)

    # Plot the confusion matrix
    if plotConfusion:
        plot_confusion_matrix(matrix, categories)

# Get images, labels tuple for CK+ datset
def importCKPlusDataset(dir = 'CKPlus', categories = None, includeNeutral = False, contemptAs = None):
    ############################################################################
    # Function: importCKPlusDataset
    # Depending on preferences, this ranges from 309 - 920 images and labels
    #    - 309 labeled images
    #    - 18 more "Contempt" images (not in our vocabulary)
    #    - 593 neutral images
    # 
    # For this to work, make sure your CKPlus dataset is formatted like this:
    # CKPlus = root (or whatever is in your 'dir' variable)
    # CKPlus/CKPlus_Images = Root for all image files (no other file types here)
    #    Example image path:
    #    CKPlus/CKPlus_Images/S005/001/S005_001_00000011.png
    # 
    # CKPlus/CKPlus_Labels = Root for all image labels (no other file types)
    #    Example label path:
    #    CKPlus/CKPlus_Labels/S005/001/S005_001_00000011_emotion.png
    #
    # CKPlus/* - anything else in this directory is ignored, as long as it
    # is not in the _Images or _Labels subdirectories
    # 
    # Optional inputs:
    # dir - Custom root directory for CKPlus dataset (if not 'CKPlus')
    #
    # includeNeutral - Boolean to include neutral pictures or not
    #    Note: Every sequence begins with neutral photos, so neutral photos
    #    greatly outnumber all other combined (approximately 593 to 327)
    #
    # contemptAs - Since it's not in our vocabulary, by default all pictures
    # labeled "Contempt" are discarded. But if you put a string here, e.g.
    # "Disgust", pictures labeled "Contempt" will be lumped in with "Disgust"
    # instead of being discarded.
    #
    #
    # RETURN VALUES:
    # images, labels = List of image file paths, list of numeric labels
    # according to EitW numbers
    #
    # Author: Dan Duncan
    #
    ############################################################################

    # Note: "Neutral" is not labeled in the CK+ dataset
    categoriesCK = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

    if categories is None:
        categoriesEitW = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
    else:
        categoriesEitW = categories

    # Root directories for images and labels. Should have no other .txt or .png files present
    dirImages = dir + '/CKPlus_Images'
    dirLabels = dir + '/CKPlus_Labels'

    if contemptAs is not None:
        # Verify a valid string was supplied
        try:
            ind = categoriesEitW.index(contemptAs)
        except ValueError:
            raise ValueError("\nError in importCKPlusDataset(): contemptAs = '" + contemptAs + "' is not a valid category. Exiting.\n")

    # Get all possible label and image filenames
    imageFiles = glob.glob(dirImages + '/*/*/*.png')
    labelFiles = glob.glob(dirLabels + '/*/*/*.txt')

    # Get list of all labeled images:
    # Convert label filenames to image filenames
    # Label looks like: CK_Plus/CKPlus_Labels/S005/001/S005_001_00000011_emotion.txt
    # Image looks like: CK_Plus/CKPlus_Images/S005/001/S005_001_00000011.png
    allLabeledImages = []

    for label in labelFiles:
        img = label.replace(dirLabels,dirImages)
        img = img.replace('_emotion.txt','.png')
        allLabeledImages.append(img)

    # Construct final set of labeled image file names and corresponding labels
    # Be sure not to include images labeled as "contempt", since those are not part of our vocabulary
    labeledImages = []
    labels = []
    labelNames = []
    contemptImages = []
    for ind in range(len(labelFiles)):
        curLabel = labelFiles[ind]
        curImage = allLabeledImages[ind]

        # Open the image as binary read-only
        with open(curLabel, 'rb') as csvfile:

            # Convert filestream to csv-reading filestream
            rd = csv.reader(csvfile)
            str = rd.next()

            # Get integer label in CK+ format
            numCK = int(float(str[0]))

            # Get text label from CK+ number
            labelText = categoriesCK[numCK-1]

            if labelText != 'Contempt':
                numEitW = categoriesEitW.index(labelText)
                labeledImages.append(curImage)
                labels.append(numEitW)
                labelNames.append(labelText)
            elif contemptAs is not None:
                # Lump "Contempt" in with another category
                numEitW = categoriesEitW.index(contemptAs)
                labeledImages.append(curImage)
                labels.append(numEitW)
                labelNames.append(labelText)
            else:
                # Discard "Contempt" image
                contemptImages.append(curImage)

    if includeNeutral:
        # Add all neutral images to our list too:
        # The first image in every series is neutral
        neutralPattern = '_00000001.png'
        neutralInd = categoriesEitW.index('Neutral')
        neutralImages = []
        neutralLabels = []
        neutralLabelNames = []

        for imgStr in imageFiles:
            if neutralPattern in imgStr:
                neutralImages.append(imgStr)
                neutralLabels.append(neutralInd)
                neutralLabelNames.append('Neutral')

        # Combine lists of labeled and neutral images
        images = labeledImages + neutralImages
        labels = labels + neutralLabels
        labelNames = labelNames + neutralLabelNames

    else:
        images = labeledImages

    # For testing only:
    #images = images[0:10]
    #labels = labels[0:10]

    return images, labels #, labelNames

# Get entire dataset
# Inputs: Dataset root directory; optional dataset name
# Returns: List of all image file paths; list of correct labels for each image
def importDataset(dir, dataset, categories):
    imgList = glob.glob(dir+'/*')
    labels = None

    # Datset-specific import rules:
    if dataset.lower() == 'jaffe' or dataset.lower() == 'training':
        # Get Jaffe labels
        jaffe_categories_map = {
            'HA': categories.index('Happy'),
            'SA': categories.index('Sad'),
            'NE': categories.index('Neutral'),
            'AN': categories.index('Angry'),
            'FE': categories.index('Fear'),
            'DI': categories.index('Disgust'),
            'SU': categories.index('Surprise')
            }

        labels = []

        for img in imgList:
            if os.path.isdir(img):
                continue
            key = img.split('.')[1][0:2]
            labels.append(jaffe_categories_map[key])

    elif dataset.lower() == 'ckplus':
        # Pathnames and labels for all images
        imgList, labels = importCKPlusDataset(dir, categories=categories,includeNeutral=True,contemptAs=None)

    elif dataset.lower() == 'misc':
        labels = [0,1,2,3,4,5,6]

    else:
        print 'Error - Unsupported dataset: ' + dataset
        return None

    # Make sure some dataset was imported
    if len(imgList) <= 0:
        print 'Error - No images found in ' + str(dir)
        return None

    # Return list of filenames
    return imgList, labels

# Delete all files in a directory matching pattern
def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))

# Delete a directory
def rmdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)

# Create a directory. Overwrite any existing directories
def mkdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)


# Create a directory only if it does not already exist
def mkdirNoForce(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)




def flatten(biglist):
    return [item for sublist in biglist for item in sublist]



# Load an image whose array elements are in uint8 format
# Caffe imports images in float format...elements vary from 0.0 to 1.0
# OpenCV Webcam brings in images in uint8 format...elements range from 0 to 255
def loadUintImage(imgFile):
  img = toggleRGB(caffe.io.load_image(imgFile))
  img *= 255.0
  img = img.astype(np.uint8)
  return img

# Load an emoji according to the desired category
def loadEmoji(ind=3):
  categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
  emojiFile = 'datasets/Emojis/' + categories[ind] + '.png'
  emoji = loadUintImage(emojiFile)
  return emoji

# Load all emojis into a list
# Assumes emojis are stored in "datasets/Emojis/angry.png" etc
def loadAllEmojis(emojiDir=None, categories=None):
    if emojiDir is None:
        emojiDir = 'datasets/Emojis/'
    if categories is None:
        categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
  
    emojis = []
    for cat in categories:
        emojiFile = emojiDir + cat + ".png"
        emojis.append(loadUintImage(emojiFile))

    return emojis
    

# Suppress print statements within a function call
# To use, write this:
# with nostdout():
#    yourfunction();
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    yield
    sys.stdout = save_stdout

