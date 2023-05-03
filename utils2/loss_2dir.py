# build loss functions
#criterion1=nn.BCEWithLogitsLoss()

import torch
from torch.cuda import is_available
from torchvision.models import vgg19
from torch import nn


criterion_mse=nn.MSELoss()
criterion_L1=nn.L1Loss()
    
device= 'cuda' if is_available() else 'cpu'
def getvggmodel():
    
    vgg_model=vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features[:-1]
    for param in vgg_model.parameters():
        param.requires_grad_(False)
    vgg_model=vgg_model.to(device)
    return vgg_model
    
    
vgg_model=getvggmodel()

#def test_mse(d1,d2):
 #   return criterion_mse(d1,d2) , criterion_L1(d1,d2)

def get_gen_loss_mse(gen_fake_pred,dis_fake_pred,image_hr):
    gen_loss=criterion_mse(dis_fake_pred,torch.ones_like(dis_fake_pred))
    content_loss=criterion_mse(image_hr,gen_fake_pred)
    return 1e-3*gen_loss+content_loss

def get_gen_loss_L1(gen_fake_pred,dis_fake_pred,image_hr):
    gen_loss=criterion_mse(dis_fake_pred,torch.ones_like(dis_fake_pred))
    content_loss=criterion_L1(image_hr,gen_fake_pred)
    return 1e-3*gen_loss+content_loss

def get_gen_loss_perceptual(gen_fake_pred,dis_fake_pred,image_hr):    
    gen_loss=criterion_mse(dis_fake_pred,torch.ones_like(dis_fake_pred))
    perceptual_loss=criterion_mse(vgg_model(torch.tile(gen_fake_pred,[3,1,1])),
                                  vgg_model(torch.tile(image_hr,[3,1,1])))
    return 1e-3*gen_loss+perceptual_loss

def get_gen_loss_fft(gen_fake_pred,dis_fake_pred,image_hr):
    gen_loss=criterion_mse(dis_fake_pred,torch.ones_like(dis_fake_pred))
    
    dummy1=gen_fake_pred.squeeze()
    ls=(dummy1*torch.eye(dummy1.shape[1]).to(device)).sum(dim=0)#.numpy()
    t_lr=torch.fft.fft(ls).abs()
    
    dummy2=image_hr.squeeze()
    ls=(dummy2*torch.eye(dummy2.shape[1]).to(device)).sum(dim=0)
    t_hr=torch.fft.fft(ls).abs()

    perceptual_loss=criterion_mse(t_lr,t_hr)
    return 1e-3*gen_loss+perceptual_loss

def get_dis_loss(dis_fake_pred,dis_real_pred):
    loss=criterion_mse(dis_fake_pred,torch.zeros_like(dis_fake_pred)) + \
    criterion_mse(dis_real_pred,torch.ones_like(dis_real_pred))
    return loss/2

