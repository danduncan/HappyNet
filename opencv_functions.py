###############################################################################
# OpenCV face recognition and segmentation
#
# This file contains utility functions for using OpenCV for face detection
# and other tasks.
#
# Face detection is done with Haar Cascades, whose weights must be downloaded
# from online resources.
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

# Load Haar cascades from XML files
def load_cascades():
    # Load Haar cascade files containing features
    cascPaths = ['models/haarcascades/haarcascade_frontalface_default.xml',
                 'models/haarcascades/haarcascade_frontalface_alt.xml',
                 'models/haarcascades/haarcascade_frontalface_alt2.xml',
                 'models/haarcascades/haarcascade_frontalface_alt_tree.xml'
                 'models/lbpcascades/lbpcascade_frontalface.xml']
    faceCascades = []
    for casc in cascPaths:
        faceCascades.append(cv.CascadeClassifier(casc))

    return faceCascades

# Do Haar cascade face detection on a single image
# Face detection returns a list of faces
# Where each face is the coordinates of a rectangle containing a face:
#   (x,y,w,h)
def DetectFace(image,color,faceCascades,single_face,second_pass,draw_rects,scale=1.0):
    # Resize
    img = cv.resize(image, (0,0), fx=1, fy=1, interpolation = cv.INTER_CUBIC)

    # Convert to grayscale and equalize the histogram
    if color:
        gray_img = img.copy().astype(np.uint8)
        gray_img = cv.cvtColor(gray_img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy().astype(np.uint8)
    cv.equalizeHist(gray_img, gray_img)

    # Detect the faces
    faces = faceCascades[2].detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(50, 50),
        flags = cv.CASCADE_SCALE_IMAGE)

    # Eliminate spurious extra faces
    discardExtraFaces = False   # Set to true to enable
    if discardExtraFaces and len(faces) > 1:
        faces = faces[0,:]
        faces = faces[np.newaxis,:]

    # Rescale cropBox

    if scale != 1.0 and len(faces) > 0:
        for i in range(faces.shape[0]):
            faces[i] = rescaleCropbox(img,faces[i],scale)

    print('Detected %d faces.' % len(faces))
    # Draw a rectangle around the faces
    if draw_rects:
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # For laboratory images, remove any spurious detections
    if single_face and len(faces) > 1:
        faces = faces[0,:]
        faces = faces[np.newaxis,:]

    if len(faces) > 0 and second_pass:
        approved = []
        for i in range(len(faces)):
            cropped_face = imgCrop(gray_img, faces[i])
            alt_check = faceCascades[1].detectMultiScale(
                cropped_face,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(int(0.8*faces[i][2]), int(0.8*faces[i][3])),
                flags = cv.CASCADE_SCALE_IMAGE)
            # Check if exactly 1 face was detected in cropped image
            if len(alt_check) == 1:
                approved.append(i)
        faces = faces[approved]
        
    return img, faces

# Resize cropBox
# This is useful if you want the face rectangle to be slightly bigger
# such aqs making it the size of the person's whole head.
def rescaleCropbox(img,cropBox,scale=1.0):
    x, y, w, h = cropBox

    # Check for valid box sizes
    if scale <= 0:
        # Invalid input. Return original
        return cropBox


    if scale < 1.0:
        x += int(w*(1-scale)/2)
        y += int(h*(1-scale)/2)
        w = int(w*scale)
        h = int(h*scale)

    elif scale > 1.0:
        x -= int(w*(scale-1.0)/2)
        y -= int(h*(scale-1.0)/2)
        w = int(w*scale)
        h = int(h*scale)

        # Make sure dimensions won't be exceeded:
        exceeded = False; count = 0; maxCount = 10 # Arbitrary magic number
        while True:
            if x < 0:
                w += 2*x # Make w smaller to maintain symmetry
                x = 0

            if y < 0:
                h += 2*y
                y = 0
                exceeded = True
            
            if x+w > img.shape[1]:
                x -= x + w - img.shape[1]
                exceeded = True
            
            if y+h > img.shape[0]:
                y -= y + h - img.shape[0]
                exceeded = True
            
            if count > maxCount:
                # Rescaling has failed. Just return original image
                print "Error: opencv_functions.imgCrop: Crop scale exceeded image dimensions"
                return cropBox

            if not exceeded:
                # Rescaling succeeded!
                break
            else:
                count += 1
                exceeded = False

    # Return rescaled cropbox
    return (x,y,w,h)


# Crop image array to pixels indicated by crop box
def imgCrop(img, cropBox, scale=1.0):
    cropBox = rescaleCropbox(img,cropBox,scale)
    (x,y,w,h) = cropBox
    img = img[y:(y+h), x:(x+h)]
    return img

# Convert bgr to rgb
# bgr is a common format and the default one for opencv
def rgb(bgr_img):
    b,g,r = cv.split(bgr_img)       # get b,g,r
    rgb_img = cv.merge([r,g,b])     # switch it to rgb
    return rgb_img

# Given directory loc, get all images in directory and crop to just faces
# Returns face_list, an array of cropped image file names
def faceCrop(targetDir, imgList, color, single_face):
    # Load list of Haar cascades for faces
    faceCascades = load_cascades()

    # Iterate through images
    face_list = []
    for img in imgList:
        if os.path.isdir(img):
            continue
        pil_img = Image.open(img)
        if color:
            cv_img  = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
        else:
            cv_img = np.array(pil_img)
            # Convert to grayscale if this image is actually color
            if cv_img.ndim == 3:
                cv_img = cv.cvtColor(np.array(pil_img), cv.COLOR_BGR2GRAY)

        # Detect all faces in this image
        scaled_img, faces = DetectFace(cv_img, color, faceCascades, single_face, second_pass=False, draw_rects=False)

        # Iterate through faces
        n=1
        for face in faces:
            cropped_cv_img = imgCrop(scaled_img, face, scale=1.0)
            if color:
                cropped_cv_img = rgb(cropped_cv_img)
            fname, ext = os.path.splitext(img)
            cropped_pil_img = Image.fromarray(cropped_cv_img)
            #save_name = loc + '/cropped/' + fname.split('/')[-1] + '_crop' + str(n) + ext
            save_name = targetDir + '/' + fname.split('/')[-1] + '_crop' + str(n) + ext
            cropped_pil_img.save(save_name)
            face_list.append(save_name)
            n += 1

    return face_list

# Add an emoji to an image at a specified point and size
# Inputs: img, emoji are ndarrays of WxHx3
#         faces is a list of (x,y,w,h) tuples for each face to be replaced
def addEmoji(img,faces,emoji):
  for x,y,w,h in faces:
    # Resize emoji to desired width and height
    dim = max(w,h)
    em = cv.resize(emoji, (dim,dim), interpolation = cv.INTER_CUBIC)

    # Get boolean for transparency
    trans = em.copy()
    trans[em == 0] = 1
    trans[em != 0] = 0

    # Delete all pixels in image where emoji is nonzero
    img[y:y+h,x:x+w,:] *= trans

    # Add emoji on those pixels
    img[y:y+h,x:x+w,:] += em

  return img

# Add emojis to image at specified points and sizes
# Inputs: img is ndarrays of WxHx3
#         emojis is a list of WxHx3 emoji arrays
#         faces is a list of (x,y,w,h) tuples for each face to be replaced
#         Labels is a list of integer labels for each emotion
def addMultipleEmojis(img,faces,emojis,labels):
    categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
    
    for i in range(len(labels)):

        x,y,w,h = faces[i]
        label = labels[i]
        emoji = emojis[int(label)]


        # Resize emoji to desired width and height
        dim = max(w,h)
        em = cv.resize(emoji, (dim,dim), interpolation = cv.INTER_CUBIC)

        # Get boolean for transparency
        trans = em.copy()
        trans[em == 0] = 1
        trans[em != 0] = 0

        # Delete all pixels in image where emoji is nonzero
        img[y:y+h,x:x+w,:] *= trans

        # Add emoji on those pixels
        img[y:y+h,x:x+w,:] += em

    return img


# Switch between RGB and BGR
def toggleRGB(img):
  r,g,b = cv.split(img)
  img = cv.merge([b,g,r])
  return img


# Combine two images for displaying side-by-side
# If maxSize is true, crops sides of image to keep under 2880 pixel width of screen
def cvCombineTwoImages(img1,img2,buf=2,maxSize=True):
  h1, w1, c1 = img1.shape
  h2, w2, c2 = img2.shape

  # Choose video size. Can be sized to either maximum screen size, or to the size of a YouTube video
  if maxSize == True:
    maxType = 'youtube'

    if maxType == 'youtube':
      # Convert to a 16:9 aspect ratio (YouTube's native aspect ratio)
      wh = 16.0/9.0 # = 1.778

      h = max(h1,h2)
      maxWidth = int(wh*float(h))
      excess = w1 + w2 + buf - maxWidth

    elif maxType == 'screen':
      screenWidth = 1920 # Width in pixels for macbook pro is 2880
      margin = 40 # Minimum number of extra pixels to save
      excess = w1 + w2 + buf - screenWidth + margin


    diff = int(np.ceil(float(excess)/4.0))

    img1 = img1[:,diff:-diff,:]
    img2 = img2[:,diff:-diff,:]

    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape

    #print "\tImages resized. New combined width: " + str(w1 + w2 + buf)

  h = max(h1,h2)
  w = w1 + w2 + buf
  c = max(c1,c2)

  if c1 != c2:
    # Incompatible dimensions
    print "Error, images have imcompatible dimensions along depth axis"
    return None

  img = np.zeros([h,w,c]).astype(np.uint8)

  # Add in the two images
  img[0:h1,0:w1,:] = img1
  img[0:h2,w1+buf:w1+buf+w2,:] = img2

  # Returned combined image as numpy array of uint8's
  return img


# Create a directory only if it does not already exist
def mkdirNoForce(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


# Save a test image with a default name from the current timestamp
def saveTestImage(img,filename=None,outDir=None):
  # Get image filename from current timestamp
  if filename is None:
    ts = time.time()
    formatStr = "%Y-%m-%d_%H-%M-%S"
    filestr = datetime.datetime.fromtimestamp(ts).strftime(formatStr)
    filename = filestr + ".png"

  if outDir is not None:
    mkdirNoForce(outDir)
    filename = outDir + "/" + filename

  # Save image
  im = Image.fromarray(toggleRGB(img))
  im.save(filename)

  # Return filename
  return filename
