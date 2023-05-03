#!/usr/bin/env python
# coding: utf-8

# In[2]:

import numpy as np
import os
import glob
from tqdm import tqdm
import random
import cv2
from scipy.interpolate import UnivariateSpline


class Filters:        
    @staticmethod
    #brightness adjustment
    def Bright(img,beta_value):
        img_bright = cv2.convertScaleAbs(img, beta=beta_value)
        return img_bright

    #summer effect
    def Summer1(img):
        increaseLookupTable = Filters._LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = Filters._LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel,red_channel  = cv2.split(img)
        red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
        sum2= cv2.merge((blue_channel, green_channel, red_channel ))
        return sum2

    #summer effect
    def Summer2(img):
        increaseLookupTable = Filters._LookupTable([0, 64, 128, 256], [0, 70, 140, 256])
        decreaseLookupTable = Filters._LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel,red_channel  = cv2.split(img)
        red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
        sum1= cv2.merge((blue_channel, green_channel, red_channel ))
        return sum1

    #winter effect
    def Winter1(img):
        increaseLookupTable = Filters._LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = Filters._LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel,red_channel = cv2.split(img)
        red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
        win1= cv2.merge((blue_channel, green_channel, red_channel))
        return win1

    #winter effect
    def Winter2(img):
        increaseLookupTable = Filters._LookupTable([0, 64, 128, 256], [0, 70, 140, 256])
        decreaseLookupTable = Filters._LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel,red_channel = cv2.split(img)
        red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
        win2= cv2.merge((blue_channel, green_channel, red_channel))
        return win2

    #blur effect
    def Blur(img,mode='avg',filter=(5,5)):
        if mode=='avg':
            blur_img = cv2.blur(img,filter)
        else: 
            blur_img = cv2.GaussianBlur(img, filter,0) 
        return blur_img

    #HDR effect
    def HDR(img):
        hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
        return  hdr

    #sharp effect
    def Sharpen1(img):
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        img_sharpen = cv2.filter2D(img, -1, kernel)
        return img_sharpen

    #sharp effect
    def Sharpen2(img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img_sharpen = cv2.filter2D(img, -1, kernel)
        return img_sharpen
    
    def Add_pos_noise(img):
        dummy=img.astype('float')+np.random.choice(np.arange(1,11), size=1, replace=False)*np.random.randn(*img.shape)
        return dummy.astype('uint8')
    
    def Add_neg_noise(img):
        dummy=img.astype('float')-np.random.choice(np.arange(1,11), size=1, replace=False)*np.random.randn(*img.shape)
        return dummy.astype('uint8')
    
    def _LookupTable(x, y):
        spline = UnivariateSpline(x, y)
        return spline(range(256))