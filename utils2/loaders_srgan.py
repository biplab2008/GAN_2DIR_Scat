#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np



# In[2]:


class MicroscopyData_old(Dataset):
    def __init__(self,
                 path=r'E:\pys\AI_algos\GAN\datasets\Protein_Atlas\Minibatch',
                 id_file='selected_files.npy',
                 transform_hr=transforms.Compose([transforms.ToTensor()]),
                 transform_lr=transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((64,64),interpolation=3),
                                                  transforms.ToTensor(),
                                                  #transforms.Normalize((0,0,0),(0.5,0.5,0.5))
                                              ]),
                 mode='train'
                ):
        '''
        load microscopy data & transform to lr
        # path : directory
        # id_file: id file 
        # transform_hr: list of transforms for HR/original images
        # transform_lr: list of transforms for low res images
        
        '''
        super(MicroscopyData,self).__init__()
        self.path=path
        ids=np.load(os.path.join(path,id_file),allow_pickle=True)
        if mode=='train': self.ids=ids[:int(len(ids)*0.8)]
        if mode=='test': self.ids=ids[int(len(ids)*0.8):]
                                       
        self.transform_hr=transform_hr
        self.transform_lr=transform_lr
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):
        basename=self.ids[idx]
        image=self.get_data(basename,self.path)
        
        #print(image.shape)
        #plt.imshow(image)
        #print(type(image))
        if self.transform_hr: 
            image_hr=self.transform_hr(image)
        else:
            image_hr=image
        image_lr=self.transform_lr(image_hr)
        
        return (image_hr,image_lr)
    
    def get_data(self,
                 basename=None,
                 trg=r'E:\pys\AI_algos\GAN\datasets\Protein_Atlas\Minibatch'):
    
        #basenames=np.load(os.path.join(trg,'selected_files.npy'),allow_pickle=True)
        # note .copy() is used to copy the data and keep the orginal data
        c1=np.array(Image.open(os.path.join(trg,basename+'_red.png')),dtype=np.float32)
        c2=np.array(Image.open(os.path.join(trg,basename+'_green.png')),dtype=np.float32)
        c3=(np.array(Image.open(os.path.join(trg,basename+'_blue.png')),dtype=np.float32)+
        np.array(Image.open(os.path.join(trg,basename+'_yellow.png')),dtype=np.float32))/2
        #image=torch.Tensor(np.stack((c1,c2,c3),axis=-1)/255).permute(2,1,0)
        image=np.stack((c1,c2,c3),axis=-1)/255
        #print(image.shape)
        return image
    
    def gets(self,idx):
        basename=self.ids[idx]
        image=self.get_data(basename,self.path)
        return image
        
        

def patchify_batch(image_tensor):
    batch_size = image_tensor.shape[0]
    
    patch_agg=[]
    for k in range(batch_size):
        dummy=patchify(image_tensor[k])
        patch_agg.extend(dummy)
        
    return torch.Tensor(patch_agg)

def patchify(x,verbose=False):
    kernel_size, stride = x.shape[1]//2,x.shape[2]//2
    patches = x.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
    patches = patches.contiguous().view(patches.size(0), -1, kernel_size, kernel_size)
    #print(patches.shape) # channels, patches, kernel_size, kernel_size

    if verbose:
        plt.figure(figsize=(18,10))
        plt.subplot(2,3,1)
        plt.imshow(x.permute(1,2,0));plt.axis('off')
        plt.subplot(2,3,2)
        plt.imshow(patches[:,0,:,:].squeeze().permute(1,2,0));plt.axis('off')
        plt.subplot(2,3,3)
        plt.imshow(patches[:,1,:,:].squeeze().permute(1,2,0));plt.axis('off')
        plt.subplot(2,3,4)
        plt.imshow(x.permute(1,2,0));plt.axis('off')
        plt.subplot(2,3,5)
        plt.imshow(patches[:,2,:,:].squeeze().permute(1,2,0));plt.axis('off')
        plt.subplot(2,3,6)
        plt.imshow(patches[:,3,:,:].squeeze().permute(1,2,0));plt.axis('off')
      
    patches_agg=[patches[:,k,:,:].squeeze().tolist() for k in range(4)]
    return patches_agg

class MicroscopyData(Dataset):
    def __init__(self,
                 path=r'E:\pys\AI_algos\GAN\datasets\Protein_Atlas\Minibatch',
                 id_file='selected_files.npy',
                 transform_hr=transforms.Compose([transforms.ToTensor()]),
                 transform_lr=transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((64,64),interpolation=3),
                                                  transforms.ToTensor(),
                                                  #transforms.Normalize((0,0,0),(0.5,0.5,0.5))
                                              ]),
                 mode='train'
                ):
        '''
        load microscopy data & transform to lr
        # path : directory
        # id_file: id file 
        # transform_hr: list of transforms for HR/original images
        # transform_lr: list of transforms for low res images
        
        '''
        super(MicroscopyData,self).__init__()
        self.path=path
        ids=np.load(os.path.join(path,id_file),allow_pickle=True)
        if mode=='train': self.ids=ids[:int(len(ids)*0.8)]
        if mode=='test': self.ids=ids[int(len(ids)*0.8):]
                                       
        self.transform_hr=transform_hr
        self.transform_lr=transform_lr
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):
        basename=self.ids[idx]
        image=self.get_data(basename,self.path)
        
        #print(image.shape)
        #plt.imshow(image)
        #print(type(image))
        if self.transform_hr: 
            image_hr=self.transform_hr(image)
        else:
            image_hr=image
        image_lr=self.transform_lr(image_hr)
        
        image_patch_hr=self.patchify(image_hr)
        image_patch_lr=self.patchify(image_lr)
        
        #image_patch_hr=image_hr
        #image_patch_lr=image_lr
        
        return (image_patch_hr,image_patch_lr)
    
    def get_data(self,
                 basename=None,
                 trg=r'E:\pys\AI_algos\GAN\datasets\Protein_Atlas\Minibatch'):
    
        #basenames=np.load(os.path.join(trg,'selected_files.npy'),allow_pickle=True)
        # note .copy() is used to copy the data and keep the orginal data
        c1=np.array(Image.open(os.path.join(trg,basename+'_red.png')),dtype=np.float32)
        c2=np.array(Image.open(os.path.join(trg,basename+'_green.png')),dtype=np.float32)
        c3=(np.array(Image.open(os.path.join(trg,basename+'_blue.png')),dtype=np.float32)+
        np.array(Image.open(os.path.join(trg,basename+'_yellow.png')),dtype=np.float32))/2
        #image=torch.Tensor(np.stack((c1,c2,c3),axis=-1)/255).permute(2,1,0)
        image=np.stack((c1,c2,c3),axis=-1)/255
        #print(image.shape)
        return image
        
    def patchify(self,x):
        kernel_size, stride = x.shape[1]//2,x.shape[2]//2
        patches = x.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        patches = patches.contiguous().view(patches.size(0), -1, kernel_size, kernel_size)
        #print(patches.shape) # channels, patches, kernel_size, kernel_size

        patches_agg=[patches[:,k,:,:].squeeze().tolist() for k in range(4)]
        return torch.Tensor(patches_agg)
