#!/usr/bin/env python
# coding: utf-8

# ### LinearGAN
# GAN using MNIST database. Usage of fully-connected dense network.
# 
# ### Notes::
# 1.   retain_graph=True, otherwise next iteration graph gets disconnected!
# 2.   batchnorm=False for Discriminator, whereas True for generator
# 3.   nn.ModuleList() instead of list of layers
# 4.   nn.LeakyReLU() in discriminator whereas nn.ReLU in generator 
# 5.   Custom initialization : No significant changes if initalization using normal distribution is used or not
# 6.   How to adaptively change learning rate like pytorch callbacks does?
# 7.   Hyperparameter Tuning in pytorch
# 8.   Create custom dataloaders for own data

# In[1]:

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from torch import nn
from torchvision.utils import make_grid

# In[15]:


## generator
class Generator(nn.Module):
    def __init__(self,
                z_dim=10,
                out_dim=28*28,
                hidden_layers=[64,128,256],
                activations=[nn.ReLU(),nn.ReLU(),nn.ReLU()],
                batchnorms=[True,True,True]):
        
        super(Generator,self).__init__()
        self.hidden_layers=hidden_layers
        self.activations=activations
        self.batchnorms=batchnorms
        self._check_dimensions()
        
        layers=nn.ModuleList()
        old_layer=z_dim
        for idx, layer in enumerate(hidden_layers):
            layers.append(nn.Linear(old_layer,layer))
            if batchnorms[idx]:layers.append(nn.BatchNorm1d(num_features=layer))
            layers.append(activations[idx])
            old_layer=layer
        
        # final layer
        layers.append(nn.Linear(hidden_layers[-1],out_dim))
        layers.append(nn.Sigmoid())
        self.layers=layers
    
    def forward(self,z):
        for layer in self.layers:
            z=layer(z)
        return z
    
    def _check_dimensions(self):
        assert len(self.hidden_layers)==len(self.activations)
        assert len(self.hidden_layers)==len(self.batchnorms)

## discriminator
class Discriminator(nn.Module):
    def __init__(self,
                in_dim=784,
                out_dim=1,
                hidden_layers=[512,256,128],
                activations=[nn.LeakyReLU(0.2),nn.LeakyReLU(0.2),nn.LeakyReLU(0.2)],
                batchnorms=[True,True,True]):
        super(Discriminator,self).__init__()
        self.hidden_layers=hidden_layers
        self.activations=activations
        self.batchnorms=batchnorms
        self._check_dimensions()
        
        layers=nn.ModuleList()
        
        old_dim=in_dim
        for idx,layer in enumerate(hidden_layers):
            layers.append(nn.Linear(old_dim,layer))
            if batchnorms[idx]:layers.append(nn.BatchNorm1d(num_features=layer))
            layers.append(activations[idx])
            old_dim=layer
        
        # final layer
        layers.append(nn.Linear(layer,out_dim))
        #layers.append(nn.Sigmoid())
        
        self.layers=layers
        
    def _check_dimensions(self):
        assert len(self.hidden_layers)==len(self.activations)
        assert len(self.hidden_layers)==len(self.batchnorms)
        
    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
        
## Noise  
def get_noise(dims=(5,10),device='cuda'):
    return torch.randn(*dims,device=device)

    
## losses
def get_discriminator_loss(criterion,
                           noise,
                           gen,
                           dis,
                           real_img):
    
    fake_img=gen(noise).detach()
    dis_fake_pred=dis(fake_img)
    dis_real_pred=dis(real_img)
        
    dis_loss_real=criterion(dis_real_pred,torch.ones_like(dis_real_pred))
    dis_loss_fake=criterion(dis_fake_pred,torch.zeros_like(dis_fake_pred))
    
    return (dis_loss_real+dis_loss_fake)/2

def get_generator_loss(criterion,
                       noise,
                       gen,
                       dis):
    fake_img=gen(noise)
    fake_pred=dis(fake_img)
    gen_loss_fake=criterion(fake_pred,torch.ones_like(fake_pred))
    return gen_loss_fake


        
# training

class training():
    def __init__(self,
                 gen,
                 dis,
                 lr=0.00002,
                 beta_1=0.5,
                 beta_2=0.999,
                 criterion=nn.BCEWithLogitsLoss(),
                 apply_init=False,
                 device='cuda'):

        self.gen_opt=torch.optim.Adam(gen.parameters(),lr=lr,betas=(beta_1,beta_2))
        self.dis_opt=torch.optim.Adam(dis.parameters(),lr=lr,betas=(beta_1,beta_2))
        self.device=device
        self.gen=gen
        self.dis=dis
        
        if apply_init:
            gen.apply(self._init_weight);
            dis.apply(self._init_weight);

        # loss functions
        self.criterion=criterion
        
        
    def fit(self,
            dataloader,
            epochs=200,
            z_dim=10):
        
        device=self.device
        criterion=self.criterion
        gen=self.gen
        dis=self.dis
        dis_opt=self.dis_opt
        gen_opt=self.gen_opt
        
        # training loop
        curr_step=0
        display_step=500
        mean_gen_loss=0
        mean_dis_loss=0
        

        for epoch in tqdm(range(epochs)):
            for real_img,_ in dataloader:
                real_img=real_img.view(-1,28*28).to(device) 

                # train discriminator
                noise=get_noise((real_img.shape[0],z_dim),device=device)
                dis_opt.zero_grad()

                dis_loss=get_discriminator_loss(criterion,
                                                noise,
                                                gen,
                                                dis,
                                                real_img)
                dis_loss.backward(retain_graph=True)
                dis_opt.step()

                mean_dis_loss+=dis_loss.detach().cpu()/display_step


                # train generator
                gen_opt.zero_grad()
                noise2=get_noise((real_img.shape[0],z_dim),device=device)

                gen_loss=get_generator_loss(criterion,
                                            noise,
                                            gen,
                                            dis
                                           )
                gen_loss.backward(retain_graph=True)
                gen_opt.step()

                mean_gen_loss+=gen_loss.detach().cpu()/display_step

                if curr_step % display_step==0 and curr_step>0:
                    image_tensor=gen(noise2).detach().cpu()
                    print(f"epoch:{epoch},step:{curr_step},generator loss:{mean_gen_loss},discriminator loss:{mean_dis_loss}")
                    self.show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28))
                    self.show_tensor_images(real_img.cpu(), num_images=25, size=(1, 28, 28))
                    mean_dis_loss=0
                    mean_gen_loss=0

                curr_step+=1
    
    # initialization
    def _init_weight(self,layer):
        if isinstance(layer,nn.Linear):
            nn.init.normal_(layer.weight,mean=0,std=0.02)
        if isinstance(layer,nn.BatchNorm1d):
            nn.init.normal_(layer.weight,mean=0,std=0.02)
            nn.init.constant_(layer.bias,0)
            
            
    def show_tensor_images(self, image_tensor, num_images=25, size=(1, 28, 28)):
        
        '''
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in an uniform grid.
        '''
        #image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.view(-1,*size)
        image_grid = make_grid(image_unflat[:num_images], nrow=5)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()






