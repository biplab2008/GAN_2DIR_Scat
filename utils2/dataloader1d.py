import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def dummyfun(f1,f2,f3,x,samples=1000, noise=False,weight=1e-1):
    dat=[np.sin(f1*np.random.rand(1)*x)+np.sin(f2*np.random.rand(1)**x)+np.sin(f3*np.random.rand(1)*x)+np.random.rand(1) for _ in range(samples)]
    dat=np.array(dat)
    dat_noise=dat+weight*np.random.rand(*dat.shape)
    print(dat.shape)
    #dat=np.concatenate((x.reshape((-1,1)),dat.T),axis=1)
    freq=np.tile(x,[samples,1]).T
    return freq, dat.T,dat_noise.T


class DummyDataset(Dataset):
    def __init__(self, 
                 noisy_data='data_noisy.csv',
                 real_data='data_real.csv', 
                 freq_data='freq.csv',
                 root_dir=None, 
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
        if root_dir:
            noisy_data=os.path.join(root_dir,noisy_data)
            real_data=os.path.join(root_dir,real_data)
            freq_data=os.path.join(root_dir,freq_data)
            
        self.df_noisy = pd.read_csv(noisy_data,header=None).to_numpy()
        self.freq=pd.read_csv(freq_data,header=None).to_numpy()
        self.df_real = pd.read_csv(real_data,header=None).to_numpy()
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df_real)

    def __getitem__(self, idx):
        
        real=np.concatenate((self.freq[:,idx].reshape(-1,1),self.df_real[:,idx].reshape(-1,1)),axis=1).T
        
        noisy=np.concatenate((self.freq[:,idx].reshape(-1,1),self.df_noisy[:,idx].reshape(-1,1)),axis=1).T
        #print(real.shape)

        if self.transform: real = self.transform(real)
        if self.transform: noisy = self.transform(noisy)

        return torch.squeeze(noisy), torch.squeeze(real)