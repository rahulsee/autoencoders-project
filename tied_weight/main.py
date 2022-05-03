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
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

from models import EncoderTiedWeights, DecoderTiedWeights
from mnist_utils import MNIST_Util

encoder=EncoderTiedWeights().to(device)
decoder=DecoderTiedWeights().to(device)

mnist_utility = MNIST_Util(data_dir = '/content/drive/MyDrive/dataset', batch_size=256)
train_loader = mnist_utility.get_train_loader()

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

### Define the loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.003

### Set the random seed for reproducible results
torch.manual_seed(0)
optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
train_loss = []
train_time = []

num_parameters = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
num_epochs = 20

for i in range(num_epochs):
    start_time = time.time()
    mini_batch_loss = []
    for image_batch, _ in tqdm(train_loader):
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data, encoder)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        mini_batch_loss.append(loss.detach().cpu().numpy())
    time_taken = time.time() - start_time
    epoch_loss = np.mean(mini_batch_loss)
    print(f"Epoch {i+1}/{num_epochs}, Loss = {epoch_loss}, Time: {time_taken}")
    train_loss.append(epoch_loss)
    train_time.append(time_taken)

states = {
    "encoder_state": encoder.state_dict(),
    "decoder_state": decoder.state_dict(),
    "training_loss": train_loss,
    "training_time": train_time,
    "optimizer_state": optimizer.state_dict(),
    "num_parameters": num_parameters
}

torch.save(states, "tied_ae.pt")

