
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from glob import glob
import numpy as np
import shutil
from tqdm import tqdm
import random
from numpy import genfromtxt
from torch.utils.data import SubsetRandomSampler
import pandas as pd


# In[28]:
class FFTDataset(Dataset):
    def __init__(self, 
                 root_dir=r'C:\Users\bdutta\work\Matlab\water_Matlab\nufft_data', 
                 transform=transforms.Compose([transforms.ToTensor()]), 
                 target_transform=transforms.Compose([transforms.ToTensor()])
                ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir=root_dir
        
        self.lr_path=os.path.join(root_dir,'lr')
        self.hr_path=os.path.join(root_dir,'hr')
    
        if root_dir:
            os.chdir(self.lr_path)
            self.noisy_files=glob('*.csv')
            #os.chdir(self.hr_path)
            #self.real_files=glob('*.csv')
    
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        
        f_lr=self.noisy_files[idx]
        f_hr=self._get_file(f_lr)
        
        df_noisy = pd.read_csv(os.path.join(self.lr_path,f_lr),header=None).to_numpy()
        df_real = pd.read_csv(os.path.join(self.hr_path,f_hr),header=None).to_numpy()
        
        #df_real=df_real[:,1]

        real = torch.Tensor(df_real[:,1])
        #real=real[:-1]
        noisy = torch.Tensor(df_noisy[:,1])
        #noisy = torch.Tensor(df_noisy)
            
        #return  noisy.permute(1,0),real.unsqueeze(0)#,noisy.unsqueeze(0)
        return  noisy.unsqueeze(0),real.unsqueeze(0)#,
    
    def _get_file(self,fname):
        dummy=fname.split('_')
        dummy[0]='hr'
        return '_'.join(dummy)
   


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


