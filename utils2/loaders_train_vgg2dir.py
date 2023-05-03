#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
import shutil
from tqdm import tqdm
import random
from numpy import genfromtxt
from torch.utils.data import SubsetRandomSampler


# In[28]:


class IRData(Dataset):
    def __init__(self,
                 hrpath=r'D:\All_files\Matlab\water_Matlab\data\combined',
                 transform_hr=transforms.Compose([
                                                  transforms.ToTensor()
                                              ]),
                 verbose=False
                ):
        '''
        load microscopy data & transform to lr
        # path : directory
        # id_file: id file 
        # transform_hr: list of transforms for HR/original images
        # transform_lr: list of transforms for low res images
        
        '''
        super(IRData,self).__init__()
        self.hrpath=hrpath
        
        os.chdir(hrpath)
        ids=glob.glob('*') #s2/s1 files
        self.ids=ids
        
        #if shuffle: random.shuffle(self.ids)
        
        #if mode=='train': self.ids=self.ids[:int(len(ids)*0.9)]
        #if mode=='test': self.ids=self.ids[int(len(ids)*0.1):]
    
        self.transform_hr=transform_hr
        
        self.verbose=verbose
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):
        basename=self.ids[idx]
        if self.verbose : print(basename)
        (image_lr,image_hr)=self.get_data(basename)
        
        #print(image.shape)
        #plt.imshow(image)
        #print(type(image))
        
        if self.transform_hr: 
            image_hr=self.transform_hr(image_hr)
        if self.transform_lr: 
            image_lr=self.transform_lr(image_lr)    

        return (image_lr,image_hr)
    
    def get_data(self, basename=None):
    
        x=genfromtxt(os.path.join(self.lrpath,basename), delimiter=',')
        x/=np.abs(np.max(x)-np.min(x))
        
        basename_hr=self._get_y_label(basename)
        y=genfromtxt(os.path.join(self.hrpath,basename_hr), delimiter=',')
        y/=np.abs(np.max(y)-np.min(y))

        if self.verbose:
            #print(basename)
            print(basename_hr)
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
            plt.contourf(x,20);plt.axis('off');plt.title('lr: '+basename)
            plt.subplot(1,2,2)
            plt.contourf(y,20);plt.axis('off');plt.title('hr: '+self._get_y_label(basename))
            plt.show()

        return (x,y)
    
    def _get_y_label(self,x_file):
        dummy=x_file.split('.')[:-1]
        dummy=dummy[0].split('_')
        
                #print(dummy)
        return ''.join((dummy[0],'_real_',dummy[-1],'.csv'))


# Creating data indices for training and validation splits:
# get data

def train_test_split(dataset,
                     validation_split=0.3,
                     shuffle_dataset=True,
                     batch_size=10
                    ):

    
    dataset_size=len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset :
        np.random.seed(112)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    
    return (train_loader,validation_loader)
    
def train_test_split_infer(dataset,
                     validation_split=0.3,
                     shuffle_dataset=True,
                     batch_size=10
                    ):

    
    dataset_size=len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset :
        np.random.seed(112)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    
    np.random.seed()
    d1=np.random.choice(train_indices,batch_size)
    d2=np.random.choice(val_indices,batch_size)
    
    train_sampler = torch.utils.data.Subset(dataset, d1)
    valid_sampler = torch.utils.data.Subset(dataset, d2)

    train_loader = torch.utils.data.DataLoader(train_sampler, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(valid_sampler, batch_size=batch_size, shuffle=False)
    
    return (d1,d2,train_loader,validation_loader)


def get_fs(lrpath=r'C:\Users\bdutta\work\Matlab\water_Matlab\data\scat',
           indices=None):

    os.chdir(lrpath)
    ids=glob.glob('*')
    
    lst_f1=[]
    lst_f2=[]
    
    ids=np.array(ids)[indices]
        
    for i in range(len(ids)):
    
        idx=ids[i].split('.')[0].split('_')[-1]
        lst_f1.append('sig_f1_'+idx+'.csv')
        lst_f2.append('sig_f2_'+idx+'.csv')
        
    return(ids,lst_f1,lst_f2)
        
    
    
    





