import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class PlayingCardDataset(Dataset):
    # needs to know what to do when created
    def __init__(self, data_dir, transform=None):
        # utlizie imagefolder function to obtain the data from the data directory. Handles creating labels for us
        self.data = ImageFolder(data_dir, transform)
    
    # dataloader will need to know how many examples we have in a dataset
    def __len__(self):
        return len(self.data)

    # returns an item from the dataset
    def __getitem__(self, idx):
        return self.data[idx]

    # returns the classes from the data folder
    def classes(self):
        return self.data.classes
    
# applies a sequence of image transformations to the data
# tensors are 3d arrays with more functionality compared to np arrays
transform = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor()
                                ])
dataset= PlayingCardDataset('train', transform)
image, label = dataset[6000]

# dataloader allows us to use the Dataloader class to easily iterate through the dataset in batches of 32, shuffle is pulling in random
# Tensor is transformed into a 4D array where the first element is the batch size (number of samples in each tensor)
# purpose: model trains faster when in batches
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
