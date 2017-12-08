import dicom
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import glob
import re
import cPickle as pickle

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from dicom.contrib.pydicom_PIL import show_PIL
from keras.preprocessing.image import img_to_array, load_img


def getScaledDims(currshape, ratio):
    """ Scale images with current shape to new ratio """
    w,h = currshape
    currratio = h/float(w)
    
    if currratio > ratio: # Height is larger
        new_h = int(w*ratio)
        return (w,new_h)
    else:
        new_w = int(h/ratio)
        return (new_w, h)

def cropCenter(img,cropy,cropx):
    """ Center crop images to new height and new width, specified by cropy, cropx """
    y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy, startx:startx+cropx]

def loadImage(image_path, ratio):
    """ Returns a np matrix with ratio of dimensions specified by ratio """
    img = plt.imread(image_path)
    currshape = img.shape
    cropx, cropy = getScaledDims(currshape, ratio)
    newimg = cropCenter(img,cropx,cropy)
    return newimg

def resizeImage(image, dims):
    """ Resizes the image into dimensions specified by dims and converts img to 3 channels """
    newimg = resize(image, dims)
    
    # Convert from 1 channel to 3 channels
    newimg_3d = np.empty(dims + (3,))
    for i in range(3):
        newimg_3d[:,:,i] = newimg
        
    return newimg_3d

def loadTrueClass(path, number, dims):
    """ Path: Path to .png images
        Number: Number of images to run
        Dims: (height, width) of the final images that will be fed into the model """
    
    # Crop images to ratio specified.
    ratio = dims[1]/float(dims[0])
    croppedImgs = [loadImage(image_path, ratio) for image_path in glob.glob(path)[:number]]
    
    # Resize all images to that dims
    x_True = np.array([resizeImage(img, dims) for img in croppedImgs])
    return x_True

def main():
    dirPath = '/enc_data/eddata/pacemaker'
    dataPath = os.path.join(dirPath, 'organized-data')
    pacemakerPath = os.path.join(dirPath,"png/full_image/*.png")

    allPatients = os.listdir(dataPath)
    print('Number of patients: {}'.format(len(allPatients)))
    numTrueFiles = len(glob.glob(pacemakerPath))
    print('Number of pacemaker files: {}'.format(numTrueFiles))

    x_True = loadTrueClass(pacemakerPath, 50, (225, 255))
    pickle.dump(x_True, open( "x_true.p", "wb" ) )

