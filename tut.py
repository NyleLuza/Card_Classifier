import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataset, dataloader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class PlayingCardDataset(dataset):
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