# init
# training with L1 with 1e-5
# optimizer and loss function

import numpy as np
import torch
from torch import nn, optim
from models.IR2DGAN_scat import Generator, Discriminator
from utils2.helpers import get_base_fname
from utils2.loss_2dir import get_gen_loss_mse, get_dis_loss
from time import time
from tqdm.auto import tqdm
import os

class ModelDetails():

    def __init__(self,gen,dis,gen_opt,dis_opt,book_keeping):
        self.gen=gen
        self.dis=dis
        self.gen_opt=gen_opt
        self.dis_opt=dis_opt
        self.book_keeping=book_keeping

# You initialize the weights to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


def init_model(generator_arch=[6,64,1,1],#[num_residual_block,residual_channels,upscale_factor,num_upscale_layers]
               lr = 0.00001, 
               beta_1 = 0.9, 
               beta_2 = 0.999, 
               device='cuda'):
    
    gen = Generator(num_residual_block=generator_arch[0],
                    residual_channels=generator_arch[1],
                    upscale_factor=generator_arch[2],
                    num_upscale_layers=generator_arch[3]).to(device)
    
    gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    dis = Discriminator().to(device) 
    dis_opt = optim.Adam(dis.parameters(), lr=lr, betas=(beta_1, beta_2))

    gen = gen.apply(weights_init)
    dis = dis.apply(weights_init)
    
    book_keeping={'device':device,'lr':lr,'gen_arch':generator_arch,'betas':[beta_1,beta_2]}
    
    return ModelDetails(gen,dis,gen_opt,dis_opt,book_keeping)
    
    
    
class SRGAN():

    def __init__(self,
                 generator_arch=[6,64,1,1],#[num_residual_block,residual_channels,upscale_factor,num_upscale_layers]
                 lr = 0.00001, 
                 beta_1 = 0.9, 
                 beta_2 = 0.999, 
                 device='cuda'):
                 
        gen = Generator(num_residual_block=generator_arch[0],
                    residual_channels=generator_arch[1],
                    upscale_factor=generator_arch[2],
                    num_upscale_layers=generator_arch[3]).to(device)
    
        gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
        dis = Discriminator().to(device) 
        dis_opt = optim.Adam(dis.parameters(), lr=lr, betas=(beta_1, beta_2))

        gen = gen.apply(weights_init)
        dis = dis.apply(weights_init)
    
        book_keeping={'device':device,'lr':lr,'gen_arch':generator_arch,'betas':[beta_1,beta_2]}
        
        self.gen=gen
        self.dis=dis
        self.gen_opt=gen_opt
        self.dis_opt=dis_opt
        self.book_keeping=book_keeping
        
        
        
    def train(self,
              train_loader,
              epochs=50,
              path=r'D:\All_files\pys\AI_algos\GAN\srgan_weights\2dgan\scat_test',
              lossfun=get_gen_loss_mse,
              update=10,
              verbose=True):
              
        # unwrap models
        (gen,dis,gen_opt,dis_opt)=(self.gen,self.dis,self.gen_opt,self.dis_opt)
        lr=self.book_keeping['lr']
        device=self.book_keeping['device']
        # training loop
        dis_loss_epoch=[]
        gen_loss_epoch=[]

        sz=len(train_loader)
        basename=get_base_fname()# base file name based on time stamps

        for epoch in tqdm(range(epochs)):
            start=time()
            dis_loss_tot=0
            gen_loss_tot=0
            for idx,(image_lr,image_hr) in enumerate(train_loader):
                image_hr=image_hr.to(torch.float32).to(device)
                image_lr=image_lr.to(torch.float32).to(device)

                # train discriminator
                dis_opt.zero_grad()
                dis_true_pred=dis(image_hr)
                image_fake=gen(image_lr).detach()
                dis_fake_pred=dis(image_fake)
                #dis_true_pred=vggextractor(dis_true_pred)
                #dis_fake_pred=vggextractor(dis_fake_pred)
                dis_loss=get_dis_loss(dis_fake_pred,dis_true_pred)
                dis_loss.backward(retain_graph=True)
                dis_loss_tot+=dis_loss.detach().cpu().item()
                dis_opt.step()

                #train generator
                gen_opt.zero_grad()
                gen_fake_pred=gen(image_lr)
                dis_fake_pred=dis(gen_fake_pred)
                gen_loss=lossfun(gen_fake_pred,dis_fake_pred,image_hr)
                #gen_loss=get_gen_loss_perceptual(gen_fake_pred,dis_fake_pred,image_hr)
                #gen_loss=get_gen_loss_L1(gen_fake_pred,dis_fake_pred,image_hr)
                gen_loss.backward(retain_graph=True)
                gen_loss_tot+=gen_loss.detach().cpu().item()
                gen_opt.step()

            if epoch%update==0:
                #save gen params
                fname=os.path.join(path,basename+'2dgan_'+str(epoch)+'_lr_'\
                                           +str(lr)+'.pt')
                torch.save(gen.state_dict(), fname)
                
            #save losses        
            dis_loss_epoch.append(dis_loss_tot/sz)
            gen_loss_epoch.append(gen_loss_tot/sz)
            
            # print
            if verbose:
                print('gen loss:{}'.format(gen_loss_tot/sz))
                print('dis loss:{}'.format(dis_loss_tot/sz))
            end=time()
            print("elapsed time for epoch{} is {}".format(epoch,end-start))

        # save losses
        np.savetxt(os.path.join(path,'L2_loss_lr_'+str(lr)+'.txt'),np.vstack((np.array(dis_loss_epoch),np.array(gen_loss_epoch))).T)

        # save discriminator state
        fname=os.path.join(path,basename+'2dgan_params_dis_epoch_'+str(epoch)+'_lr_'\
                                       +str(lr)+'.pt')
        torch.save(dis.state_dict(), fname)
        
        return gen_loss_epoch, dis_loss_epoch
        

            