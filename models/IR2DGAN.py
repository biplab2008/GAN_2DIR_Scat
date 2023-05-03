#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import numpy as np


# generator & discriminator
class ResidualBlock(nn.Module):
    def __init__(self,num_filters=64):
        super(ResidualBlock,self).__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=num_filters),
            nn.PReLU(),
            nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=num_filters)
        )
        
    def forward(self,x):
        return self.layers(x)+x
        
    
class Generator(nn.Module):
    def __init__(self,
                 num_residual_block=16,
                 residual_channels=64,
                 upscale_factor=1):
        super(Generator,self).__init__()
        self.num_residual_block=num_residual_block
        
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=1,
                                         out_channels=residual_channels,
                                         kernel_size=9,stride=1,
                                         padding=4),
                                 nn.PReLU())
        
        conv2=nn.ModuleList()
        for n in range(self.num_residual_block):
            #conv2.append(self._residual_block(num_filters=residual_channels))
            conv2.append(ResidualBlock(num_filters=residual_channels))
        self.conv2=conv2
        
        self.conv3=nn.Sequential(nn.Conv2d(in_channels=residual_channels,
                                      out_channels=residual_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1),
                            nn.BatchNorm2d(num_features=residual_channels))
        conv4=[]
        nchans=2 if upscale_factor==1 else upscale_factor**2
        for layer in range(3):
            conv4+=[nn.Conv2d(in_channels=residual_channels,
                              out_channels=residual_channels*nchans, 
                              kernel_size=3, 
                              stride=1, 
                              padding=1), 
                            nn.PixelShuffle(upscale_factor=upscale_factor),
                            nn.PReLU()]
        self.conv4=nn.Sequential(*conv4)
        
        #self.conv5=nn.Sequential(nn.Conv2d(in_channels=64,
        #                              out_channels=256, kernel_size=3, stride=1, padding=1), 
        #                    nn.PixelShuffle(upscale_factor=2),
        #                    nn.PReLU())
        self.conv5=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=1,kernel_size=9,padding=4,stride=1),
                                 nn.Tanh())
                        
    def forward(self,x):
        x=self.conv1(x)
        old_x=x
        for layer in self.conv2:
            x=layer(x)
        
        x=self.conv3(x)+old_x
        
        print(x.shape)
        x=self.conv4(x)
        print(x.shape)
        x=self.conv5(x)
        print(x.shape)
        return x

class Discriminator(nn.Module):
    def __init__(self, 
                 in_channels=1,
                 filters=[64,128,256,512]):
        super(Discriminator,self).__init__()
        
        layers=nn.ModuleList()
        old_filter=in_channels
        for idx,new_filter in enumerate(filters):
            layers+=[self._blocks(old_filter,new_filter,stride=1,batchnorm=(idx>0))]
            layers+=[self._blocks(new_filter,new_filter,stride=2,batchnorm=True)]
            old_filter=new_filter
        
        self.conv=layers
        
        #self.linear=nn.Sequential(nn.Linear(filters[-1],1024),
        #                          nn.LeakyReLU(0.2),
        #                          nn.Linear(1024,1),
        #                          nn.Sigmoid())
        
        self.conv2=nn.Sequential(nn.Conv2d(filters[-1],1,kernel_size=3,stride=1,padding=1),
                                 nn.Sigmoid())
        
    def _blocks(self,in_channels=64,out_channels=128,stride=1,batchnorm=True):
        layers=[]
        layers.append(nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels, 
                                              kernel_size=3,
                                              stride=stride,
                                              padding=1))
                     
        if batchnorm : layers.append(nn.BatchNorm2d(num_features=out_channels))
        layers.append(nn.LeakyReLU(0.2))
        layers=nn.Sequential(*layers)
        return layers
    
    def forward(self,x):
        for layers in self.conv:
            x=layers(x)
        #x=x.reshape(-1,x.shape[1]*x.shape[2]*x.shape[3])
        #print(x.shape)
        x=self.conv2(x)

        return x


# In[2]:


gen=Generator()
z=torch.randn(2,1,16,16)
z=gen(z.to(torch.float32))
print(z.shape)
d=Discriminator()
d(z).shape


# In[ ]:




