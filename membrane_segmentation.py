#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 08:33:32 2017

@author: cedar

This script is to test different segmenting methods for cells.
"""

#%% import libs
import cv2
from matplotlib import pyplot as plt
import numpy as np

def see(img_arr):
    plt.imshow(img_arr, cmap = 'Greys')
    plt.show()
#%% Load image
src = cv2.imread('a.jpg', 0) # 0 for greyscale

plt.imshow(src); plt.show()
#%% segment using adaptive threshold
dst = cv2.adaptiveThreshold(src = src,
                                    maxValue = 255,
                                    adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C,
                                    thresholdType = cv2.THRESH_BINARY,
                                    blockSize = 3,
                                    C = 0)
see(dst)
#Does not work. Produces random image
#%% Segmentation using regular thresholding

#%% Edge preserving Smooth
blur = cv2.bilateralFilter(src,3,50,50)
see(blur)

#%% Canny
edges = cv2.Canny(blur, 80, 100)
see(edges)
#%% Morph on canny
def kernel(l):
    return cv2.getStructuringElement(cv2.MORPH_CROSS,(l,l))
morph = cv2.dilate(edges, kernel(3))
morph = cv2.erode(morph, kernel(3))
see(morph)
#%% Laplacian
laplc = cv2.Laplacian(blur, cv2.CV_16S)
see(laplc)
#%%Canny of Laplacian
cannyIn = np.uint8((laplc+256)/4)
edges = cv2.Canny(cannyIn, 30, 50)
see(edges)
#%% adaptive Threshold of blur
threshed = cv2.adaptiveThreshold(blur, 90, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize = 7, C = 7)
see(threshed)
#%% erode threshed
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
erosion = cv2.erode(threshed,kernel,iterations = 1)
see(dilation)

#%% Try scipy
from skimage import graph, data, io, segmentation, color
from skimage.measure import regionprops
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np

image = io.imread('./blur.jpg')