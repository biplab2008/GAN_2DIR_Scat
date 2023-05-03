#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


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

class VisiaData(Dataset):
    def __init__(self,
                 s2_path=r'E:\pys\AI_algos\GAN\datasets\Faces\s1_plus_s2_t2',
                 pp_path=r'E:\pys\AI_algos\GAN\datasets\Faces\pp',
                 transform_s2=transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((1024,1024),interpolation=3),
                                                  transforms.ToTensor(),
                                                  #transforms.Normalize((0,0,0),(0.5,0.5,0.5))
                                              ]),
                 transform_pol=transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((1024,1024),interpolation=3),
                                                  transforms.ToTensor(),
                                                  #transforms.Normalize((0,0,0),(0.5,0.5,0.5))
                                              ]),
                 mode='train',
                 shuffle=True,
                 verbose=False
                ):
        '''
        load microscopy data & transform to lr
        # path : directory
        # id_file: id file 
        # transform_hr: list of transforms for HR/original images
        # transform_lr: list of transforms for low res images
        
        '''
        super(VisiaData,self).__init__()
        self.s2_path=s2_path
        self.pp_path=pp_path
        os.chdir(s2_path)
        ids=glob.glob('*') #s2/s1 files
        if mode=='train': self.ids=ids[:int(len(ids)*0.9)]
        if mode=='test': self.ids=ids[int(len(ids)*0.1):]
        
        if shuffle: random.shuffle(self.ids)
        self.transform_s2=transform_s2
        self.transform_pol=transform_pol
        
        self.verbose=verbose
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):
        basename=self.ids[idx]
        if self.verbose : print(basename)
        (image_s2,image_pol)=self.get_data(basename)
        
        #print(image.shape)
        #plt.imshow(image)
        #print(type(image))
        if self.transform_s2: 
            image_s2=self.transform_s2(image_s2)
            #image_pol=self.transform_s2(image_pol)
            image_pol=self.transform_pol(image_pol)
        else:
            image_s2=image_s2
            image_pol=image_pol
            
        '''
        image_lr=self.transform_lr(image_hr)
        
        image_patch_hr=self.patchify(image_hr)
        image_patch_lr=self.patchify(image_lr)
        '''    
        image_patch_s2=self.patchify(image_s2)
        image_patch_pol=self.patchify(image_pol)
        
        
        return (image_patch_s2,image_patch_pol)#(image_s2,image_pol)#(image_patch_hr,image_patch_lr)(image_patch_s2,image_patch_pol)#
    
    def get_data(self, basename=None):
    
        x=np.array(Image.open(os.path.join(self.s2_path,basename)),dtype=np.uint8)
        y=np.array(Image.open(os.path.join(self.pp_path,self._get_y_label(basename))),dtype=np.uint8)

        if self.verbose:
            plt.figure(figsize=(6,4))
            plt.subplot(1,2,1)
            plt.imshow(x);plt.axis('off');plt.title(basename.split('.')[0])
            plt.subplot(1,2,2)
            plt.imshow(y);plt.axis('off');plt.title('polarized')
            plt.show()

        return (x,y)

    def patchify(self,x):
        kernel_size, stride = x.shape[1]//2,x.shape[2]//2
        patches = x.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        patches = patches.contiguous().view(patches.size(0), -1, kernel_size, kernel_size)
        #print(patches.shape) # channels, patches, kernel_size, kernel_size

        patches_agg=[patches[:,k,:,:].squeeze().tolist() for k in range(4)]
        return torch.Tensor(patches_agg)
    
    def _get_y_label(self,x_file):
        dummy=x_file.split('_')[:3]
        return ''.join([dummy[0],'_',dummy[1],'_',dummy[2],'.jpg'])


# In[ ]:




