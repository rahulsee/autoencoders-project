import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import time

class EncoderTiedWeights(nn.Module):
    def __init__(self):
        super(EncoderTiedWeights, self).__init__()
        self.batch_norm_1 = nn.BatchNorm2d(8)
        self.batch_norm_2 = nn.BatchNorm2d(16)
        self.encoder_module = nn.ModuleList([
            nn.Conv2d(1, 8, 3, stride=2, padding=1),   # 0
            nn.SELU(), # 1
            nn.Conv2d(8, 16, 3, stride=2, padding=1), #2
            self.batch_norm_2, #3
            nn.SELU(), #4
            nn.Conv2d(16, 32, 3, stride=2, padding=0), #5
            nn.SELU(), #6
            nn.Flatten(start_dim=1), #7
            nn.Linear(3 * 3 * 32, 128), #8
            nn.SELU(), #9
            nn.Linear(128, 8) #10
        ])

        self.bias = nn.ParameterList([
            nn.Parameter(torch.randn(1)),
            nn.Parameter(torch.randn(8)),
            nn.Parameter(torch.randn(16)),
            nn.Parameter(torch.randn(1)),
            nn.Parameter(torch.randn(1))
        ])

    def forward(self, x):
        for i in range(11):
            x=self.encoder_module[i](x)
        return x

class DecoderTiedWeights(nn.Module):
    def __init__(self):
        super(DecoderTiedWeights, self).__init__()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))


    def forward(self, x, encoder):
        x  = nn.functional.linear(x, weight=encoder.encoder_module[10].weight.transpose(0,1), bias=encoder.bias[4])
        x = nn.functional.selu(x)
        x = nn.functional.linear(x, weight=encoder.encoder_module[8].weight.transpose(0,1), bias=encoder.bias[3])
        x = nn.functional.selu(x)
        x = self.unflatten(x)
        x = nn.functional.conv_transpose2d(x, stride=2, output_padding=0, weight=encoder.encoder_module[5].weight, bias=encoder.bias[2])
        x = nn.functional.selu(x)
        x = nn.functional.conv_transpose2d(x, stride=2, padding=1, output_padding=1, weight=encoder.encoder_module[2].weight, bias=encoder.bias[1])
        x = nn.functional.selu(x)
        x = nn.functional.conv_transpose2d(x, stride=2, padding=1, output_padding=1, weight=encoder.encoder_module[0].weight, bias=encoder.bias[0])
        x = nn.functional.sigmoid(x)
        return x

