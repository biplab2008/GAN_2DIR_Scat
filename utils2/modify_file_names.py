#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[6]:


import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import numpy as np
from tqdm import tqdm
import random

def modify_pp_filename(pp_path=r'E:\pys\AI_algos\GAN\datasets\Faces\pp'):
    # run only once
    # change all file names in pp & cp to without index/Parallel_Polarized_Capture
    os.chdir(pp_path)
    pp_list=glob.glob('*')

    for k in pp_list:
        if len(k.split('_'))>4:
            #print('old:',k)
            dummy=k.split('_')
            os.rename(k,''.join([dummy[0],'_',dummy[1],'_',dummy[3],'.jpg']))
            #print('new:',''.join([dummy[0],'_',dummy[1],'_',dummy[3],'.jpg']))
            
def modify_s2_filename(s2_path=r'E:\pys\AI_algos\GAN\datasets\Faces\s1_plus_s2_t2'):
    # run only once
    # change all file names in s1&s2 to without index/Standard_Capture
    os.chdir(s2_path)
    s2_list=glob.glob('*')

    for k in s2_list:
        if len(k.split('_'))>4:
            #print('old:',k)
            dummy=k.split('_')
            os.rename(k,''.join([dummy[0],'_',dummy[1],'_',dummy[3],'_',dummy[4],'.jpg']))
            #print('new:',''.join([dummy[0],'_',dummy[1],'_',dummy[3],'.jpg']))
            
def get_y_label(x_file):
    dummy=x_file.split('_')[:3]
    return ''.join([dummy[0],'_',dummy[1],'_',dummy[2],'.jpg'])
       
    
def get_data(s2_path=r'E:\pys\AI_algos\GAN\datasets\Faces\s1_plus_s2_t2',
             pp_path=r'E:\pys\AI_algos\GAN\datasets\Faces\pp',
             basename=None,
             verbose=False):
    
    x=np.array(Image.open(os.path.join(s2_path,basename)),dtype=np.uint8)
    y=np.array(Image.open(os.path.join(pp_path,get_y_label(basename))),dtype=np.uint8)
    
    if verbose:
        plt.subplot(1,2,1)
        plt.imshow(x.astype('uint8'));plt.axis('off');plt.title(basename.split('.')[0])
        plt.subplot(1,2,2)
        plt.imshow(y.astype('uint8'));plt.axis('off');plt.title('polarized')
    
    return (x,y)


# In[ ]:




