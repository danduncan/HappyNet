#############################################################################################
#
# This is a program used to generate your own custom dataset of labeled emotions.
# It uses the webcam and prompts the user to make an emotion.
# When the user is ready, they press ENTER, and the webcam saves a snapshot of their emotion.
# All files are saved using the Japanese Female Facial Expressions (JAFFE) dataset naming 
# convention.
#
# Instructions:
# - Execute as python script
# - If working properly, a window will pop up with a feed from your webcam
#      Note: This does not appear to work on virtual machines. It was used on a Macbook.
# - Follow the prompts on the shell screen
# - When pressing a button, make sure the video screen is selected, not the text screen.
#   The video screen is the one taking your text inputs
#
# Possible text inputs:
#   ENTER - save image
#   SPACE - Skip to next emotion without saving
#   ESC   - Quit the program
#
#
# Date modified: March 2016
#
# Authors:    Dan Duncan    
#             Gautam Shine
#
#############################################################################################

import os, shutil, sys, time, re, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import Image
import caffe
import contextlib, cStringIO
import random

from caffe_functions import *
from opencv_functions import *
from utility_functions import *
from PIL import Image

#############################################################################################
#
# HELPER FUNCTIONS
#
#############################################################################################

# Filename format looks like:
# training_data/0000000.HA.0.png
# outDir is self-explanatory
# counter gets converted into string of length 7, with leading 0's
# Label is the two-character emotion label
# subCount is a number label for jittered images
# Extension is the filetype
def getFilename(counter,subCount=0,outDir=None,strLength=7,label='HA',extension='.png'):
  if outDir is None:
    outDir = ""
  else:
    outDir += '/'

  if subCount is None:
    subCount = ""
  else:
    subCount = "." + str(subCount)

  numStr = str(counter).zfill(strLength)

  return outDir + numStr + "." + label + subCount + extension 

# Suppress print statements within a function call
# Just call:
# with nostdout():
#    yourfunction();
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    yield
    sys.stdout = save_stdout

# Get a randon emotion label
def getRandomLabel(pickFrom=None):
  if pickFrom is None:
    pickFrom = [0,1,2,3,4,5,6]

  return random.choice(pickFrom)


# Crop image and save to file
def saveSingleImage(frame,file):
  # Save cropped image. Can also rescale cropbox
  im = Image.fromarray(toggleRGB(frame))
  im.save(file)

# Crop and save image, including adding jitter
def saveAcceptedImage(frame,faces,counter,outDir=None,strLength=7,label='HA',extension='.png',jitter=False):
  

  if jitter:
    frames = jitterImage(frame,faces)
  else:
    frames = [imgCrop(frame,faces[0])]

  subCount = 0
  for frame in frames:
    filename = getFilename(counter,outDir=outDir,subCount=subCount,strLength=strLength,label=suf,extension=extension)
    saveSingleImage(frame,filename)
    subCount += 1


# Jitter an image
# Returns several jittered versions of the input image
def jitterImage(frame,faces):
  # Define constants
  numShiftMax = 4;  # Number of shifted images to produce
  numColorMax = 6;  # Number of color-shifted images to produce
  maxShift = 0.1 # Maximum pixel displacement in x and y directions
  maxColorShift = 30; # Raw pixel shift

  # Frame width and height
  fw = frame.shape[1]
  fh = frame.shape[0]

  x,y,w,h = faces[0]

  frames = []; # Will hold output jittered images

  # Return original unjittered image
  frames.append(frame[y:y+h,x:x+h])

  # Shift image by up to 10% of cropbox size in each direction
  shiftCount = 0
  while shiftCount < numShiftMax:
    # Generate shifts:    -0.1 < shift < .1
    xshift = np.random.uniform(0.0,maxShift*2) - maxShift
    yshift = np.random.uniform(0.0,maxShift*2) - maxShift

    # Apply shifts
    xt = x + int(xshift*w)
    yt = y + int(yshift*h)

    # Verify shifts are within limits
    if xt >= 0 and yt >= 0 and xt+w < fw and yt+h < fh:
      # New values are ok
      frames.append(frame[yt:yt+h,xt:xt+w])
      shiftCount += 1

  # Brighten or darken image uniformly
  # Raw pixel values are 0 to 255
  for i in range(numColorMax):
    shift = random.randint(0,2*maxColorShift) - maxColorShift/2
    ftmp = frame.astype(np.int) + shift
    
    # Make sure ftmp does not exceed 0 and 255
    ftmp[ftmp < 0] = 0
    ftmp[ftmp > 255] = 255

    # Add new image to output
    ftmp = ftmp.astype(np.uint8)
    frames.append(ftmp[yt:yt+h,xt:xt+w])

  return frames




################################################################################################
#
# START SCRIPT
#
#################################################################################################

# Pick mode (train or validate)
validationMode = False

# Pick output size in pixels, of all cropped images (images are all square)
imgSize = 200;
boxScale = 1.2 # Size of crop boxes (relative to original filter size)
jitter = True; # Jitter accepted images?

# Initialize all labels
categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
suffixes   = [ 'AN',     'DI',       'FE',    'HA',       'NE',       'SA',   'SU']
pickFrom   = [                        2,                  4,          5       ] # Only prompt user for emotions in this list

if validationMode:
  jitter = False
  outDir     = 'datasets/validation_images'
else:
  jitter = True
  outDir     = 'datasets/training_images'

counter    = 300 # Used to increment filenames

# Output filename configuration:
strLength = 7; # Length of output filename number string
extension = '.png' # Output file type

# Check that outDir and counter are properly initialized
print "\n"
if not os.path.exists(outDir):
  print "Output directory does not exist. Making directory"
  os.mkdir(outDir)
else:
  print "Output directory already exists"

numCheck = 1000; # Number of filenames to check before giving up
num = 0;
while True:
  strCheck = getFilename(counter,outDir=outDir,label="*")
  print "Checking: " + strCheck
  if glob.glob(strCheck):
    print "\tError: File exists. Incrementing counter"
    counter += 1
    num += 1
  else: 
    print "First valid file is: " + strCheck
    break

  if num > numCheck:
    print "ERROR: No available filename up to " + strCheck + " could be found."
    sys.exit(0)


# Set up face detection
faceCascades = load_cascades()

# Set up display window
cv.namedWindow("preview")

# Open input video steam
vc = cv.VideoCapture(0)

# Check that video stream is running
if vc.isOpened(): # try to get the first frame
  rval, frame = vc.read()
  #frame = frame.astype(np.float32)
else:
  rval = False


print "\n"  
nextEmotion = True
while rval:

  if nextEmotion: # Generate next emotion
    nextEmotion = False

    # Generate a random integer label
    intLabel = getRandomLabel(pickFrom)

    # Get emotion string and file suffyx
    emotion = categories[intLabel]
    suf = suffixes[intLabel]

    # Print prompt to user:
    print "Emotion is: " + emotion + ".\t(ENTER to capture, SPACE to skip)"

  # Read in next frame
  rval, frame = vc.read()

  # Mirror image
  frame = np.fliplr(frame)
    
  # Detect faces
  # Find all faces
  with nostdout():
    newFrame, faces = DetectFace(frame,True,faceCascades,single_face=False,second_pass=False,draw_rects=True,scale=boxScale)

  oneFace = False  
  if faces is None or len(faces) == 0:
    # Poor input: do nothing to frame
    #newFrame = frame
    pass
  elif len(faces) > 1:
    # Too many faces found
    pass
  else:
    # Just the right number of faces found
    oneFace = True

  # Show video with or without boxed face
  cv.imshow("preview", newFrame)

  # Wait for user to press key. On ESC, close program
  key = cv.waitKey(20)
  if key == 27: # ESC --> exit on ESC
    print 'ESC was pressed! Quitting...'
    break
  elif key == 32: # SPACE --> Next image
    print 'Label skipped'
    nextEmotion = True
    continue; # Break out of loop
  elif key == 13: # ENTER --> Accept image
    if not oneFace:
      print "Error: ENTER pressed, but face invalid. Keep trying..."
      print "Emotion is: " + emotion + ".\t(ENTER to capture, SPACE to skip)"
    else:
      saveAcceptedImage(frame,faces,counter,outDir=outDir,strLength=strLength,label=suf,extension=extension,jitter=jitter)
      print 'Image accepted and saved!'
      counter += 1
      nextEmotion = True
      continue; # Break out of loop
  else: # Invalid key, ignore
    pass


cv.destroyWindow("preview")
