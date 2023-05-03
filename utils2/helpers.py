#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# In[2]:


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    #image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.figure(figsize=(10,6))
    #plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.contourf(torch.mean(image_grid.permute(1, 2, 0),axis=2))
    plt.axis('off')
    plt.show()
    
def get_base_fname():
    month = {1:'Janauary',2:'February',3:'March',4:'April',5:'May',6:'June',
             7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
    dt=datetime.today()
    return month[dt.month]+str(dt.day)+'_'+str(dt.hour)+str(dt.minute)+str(dt.second)+'_'


# In[ ]:




