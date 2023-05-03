#!/usr/bin/env python
# coding: utf-8

# ### Dataloaders

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import cuda


# load mnist data
def load_mnist(transform_list=[transforms.ToTensor()],
              mnist_path='E:/pys/AI_algos/GAN',
              shuffle=True,
              batch_size=128,
              download=False):
    #download=False if mnist_path~=None else True
    transform=transforms.Compose(transform_list)#,transforms.Normalize((0.5,),(0.5,))])
    dataloader=DataLoader(MNIST(mnist_path,download=download,transform=transform),
               batch_size=batch_size,
               shuffle=shuffle)
    return dataloader

# choose device
def get_device():
    device='cuda' if cuda.is_available() else 'cpu'
    return device

