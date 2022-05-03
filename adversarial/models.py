import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch.autograd import Variable
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def reparam(mu,logvar):
  std = torch.exp(logvar/2)
  z =  Variable(Tensor(np.random.normal(0, 1, (mu.size(0), 8)))).to(device)
  return mu + z*std

class EncoderAdv(nn.Module):
    
    def __init__(self):
        super(EncoderAdv, self).__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.SELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.SELU()
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.SELU(),
            nn.Linear(128, 8)
        )

        self.encoder_lin_2 = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.SELU(),
            nn.Linear(128, 8)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        mu = self.encoder_lin(x)
        sigma = self.encoder_lin_2(x)
        return reparam(mu, sigma)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.dropout(self.fc1(x), p=0.2)
        x = F.selu(x)
        x = F.dropout(self.fc2(x), p=0.2)
        x = F.selu(x)
        return F.sigmoid(self.fc3(x))      

class DecoderAdv(nn.Module):
    
    def __init__(self):
        super(DecoderAdv, self).__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(8, 128),
            nn.SELU(),
            nn.Linear(128, 3 * 3 * 32),
            nn.SELU()
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.SELU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.SELU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x