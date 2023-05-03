#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[7]:


import matplotlib.pyplot as plt
from torchvision.utils import make_grid
def show_tensor_images(image_tensor, num_images=4,nrow=2, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    #image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    #plt.show()


# In[ ]:




