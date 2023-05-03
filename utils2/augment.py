#!/usr/bin/env python
# coding: utf-8

# In[2]:

from utils.filter import Filters
import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import random
import cv2

class AugData:
    def __init__(self,
                 s2_path=r'E:\pys\AI_algos\GAN\datasets\Faces\s1_plus_s2',
                 add_noise=0):
        self.s2_path=s2_path
        os.chdir(s2_path)
        self.s2_list=glob.glob('*')
        #print(self.s2_list)
        self.add_noise=add_noise
        self.curr_filename=None
        self.curr_image=None
        self.gen_aug_data()
        
        
    def gen_aug_data(self):
        
        for fname in tqdm(self.s2_list):
            print('processing :'+fname)
            self.curr_filename=fname
            self.curr_image=cv2.imread(fname)
        
            self.gen_bright_images()
            self.gen_sharp_images()
            self.gen_hdr_images()
            self.gen_summer_images()
            self.gen_winter_images()
            self.gen_blur_images()
            self.gen_noisy_images()
            
    def gen_bright_images(self):
        
        choices=np.random.choice(60,5,replace=False)
        for choice in choices:
            dummy=Filters.Bright(self.curr_image,choice)
            self.save_file(dummy,'bright'+str(choice))
            
        choices=-np.random.choice(30,5,replace=False)
        for choice in choices:
            dummy=Filters.Bright(self.curr_image,choice)
            self.save_file(dummy,'dark'+str(choice))
            
    def gen_sharp_images(self):
        dummy=Filters.Sharpen1(self.curr_image)
        self.save_file(dummy,'sharp_1')
        dummy=Filters.Sharpen2(self.curr_image)
        self.save_file(dummy,'sharp_2')
        
    def gen_hdr_images(self):
        dummy=Filters.HDR(self.curr_image)
        self.save_file(dummy,'hdr')
    
    def gen_summer_images(self):
        dummy=Filters.Summer1(self.curr_image)
        self.save_file(dummy,'summer_1')
        dummy=Filters.Summer2(self.curr_image)
        self.save_file(dummy,'summer_2')
        
    def gen_winter_images(self):
        dummy=Filters.Winter1(self.curr_image)
        self.save_file(dummy,'winter_1')
        dummy=Filters.Winter2(self.curr_image)
        self.save_file(dummy,'winter_2')
        
    def gen_blur_images(self):
        dummy=Filters.Blur(self.curr_image,mode='avg',filter=(10,10))
        self.save_file(dummy,'blur_avg')
        
        dummy=Filters.Blur(self.curr_image,mode='gauss',filter=(7,7))
        self.save_file(dummy,'blur_gauss')
        
    def gen_noisy_images(self):
        dummy=Filters.Add_pos_noise(self.curr_image)
        self.save_file(dummy,'pos_noise')
        dummy=Filters.Add_neg_noise(self.curr_image)
        self.save_file(dummy,'neg_noise')
    
    def save_file(self,data,suffix):
        cv2.imwrite(self.curr_filename.split('.')[0]+'_'+suffix+'.jpg', data)


# In[ ]:




