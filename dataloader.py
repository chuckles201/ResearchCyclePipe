import torchvision
import torch
import os
from tqdm import tqdm
import glob

'''DataLoader

Here, we will be creating a custom class that
works with TorchVision, and is able to both load
images of desired shape/size, and replace them with 
latent representations, if avaialable.

'''

from torch.utils.data.dataset import Dataset

class ImageDataset(Dataset):
    def __init__(self,
                 split,
                 im_path,
                 im_size=256,
                 im_channels=3,
                 im_ext='jpg')