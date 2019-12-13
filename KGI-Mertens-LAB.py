# import the necessary packages
from imutils import paths
import skimage
import numpy as np
import os
import easygui
import argparse
import cv2
import fnmatch
import time
import sys
from imutils import contours
from skimage import measure
from skimage import morphology

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")                
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def update_progress(progress):
    barLength = 30 # Modify this to change the length of the progress bar
    
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% ".format( "#"*block + "-"*(barLength-block), int(progress*100))
    sys.stdout.write(text)
    sys.stdout.flush()

def LabMask(image, Mimage):

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    thresh = cv2.inRange(gray, 0, 80)
    
    
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
   
    alpha = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    alpha = cv2.GaussianBlur(alpha,(7,7),0)
    
    cv2.imwrite("Mask.tif",alpha)
    foreground = Mimage.astype(float)
    background = image.astype(float)
    
    Amask = alpha.astype(float)/255
 
    foreground = cv2.multiply(Amask, foreground)
 
    background = cv2.multiply(1.0 - Amask, background)
 
    return cv2.add(foreground, background)


# inout directory
dirinput = easygui.diropenbox(msg=None, title="Please select input directory", default=None)
total_con=len(fnmatch.filter(os.listdir(dirinput), '*.tif'))
msg = str(total_con) +" files do you want to continue?"
title = "Please Confirm"
if easygui.ynbox(msg, title, ('Yes', 'No')): # show a Continue/Cancel dialog
    pass # user chose Continue else: # user chose Cancel
else:
    exit(0)

# output directory
dirout = easygui.diropenbox(msg=None, title="Please select output directory", default=None)

i=0

# load the original image

for imagePath in paths.list_images(dirinput):
 i = i+1

 image = cv2.imread(imagePath)
 lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) 
 l, a, b = cv2.split(lab)
 
 # Gammas
 gamma = float(7)
 Gamma01 = np.zeros(l.shape, l.dtype)
 Gamma01 = adjust_gamma(l, gamma=gamma)
 
 lab = cv2.merge((Gamma01,a,b)) 
 imgout01 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
   
 gamma = float(0.60)
 Gamma03 = np.zeros(l.shape, l.dtype)
 Gamma03 = adjust_gamma(l, gamma=gamma)

 lab = cv2.merge((Gamma03,a,b)) 
 imgout03 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
 
 img_list = [imgout01, image, imgout03]
 merge_mertens = cv2.createMergeMertens()
 #merge_mertens.setContrastWeight(0.05)
 res_mertens = merge_mertens.process(img_list)

 # Convert datatype to 8-bit and save
 file_input = os.path.basename(imagePath)
 res_mertens_8bit = np.clip(res_mertens*255, 1, 254).astype("uint8")
 
 #cv2.imwrite(dirout+"/"+file_input,res_mertens_8bit)
 out_image = LabMask(image, res_mertens_8bit)
 cv2.imwrite(dirout+"/"+file_input,out_image)
 update_progress(i/total_con)
 
print('Done')







