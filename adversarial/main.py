from models import EncoderAdv, DecoderAdv, Discriminator
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
from tqdm import tqdm
from mnist_utils import MNIST_Util

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
encoder = EncoderAdv()
decoder = DecoderAdv()
discriminator = Discriminator()

enc_dec_parameters = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()},
]

### Define an optimizer (both for the encoder and the decoder!)
lr = 0.001
lr = 0.005

### Set the random seed for reproducible results
torch.manual_seed(0)

optim = torch.optim.Adam(enc_dec_parameters, lr=lr, weight_decay=1e-05)
optim_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)
discriminator.to(device)

reconstruction_loss_fn = torch.nn.MSELoss()
disc_loss_fn = torch.nn.BCELoss()


data_dir = 'dataset'

mnist_utility = MNIST_Util(data_dir = '/content/drive/MyDrive/dataset', batch_size=256)
train_loader = mnist_utility.get_train_loader()
num_epochs = 20
gen_losses=[]
disc_losses=[]
training_times=[]
num_parameters = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters()) + sum(p.numel() for p in discriminator.parameters())


for i in range(num_epochs):
    train_loss_g = []
    train_loss_d = []
    start_time = time.time()
    for image_batch, _ in tqdm(train_loader): 
        encoder.train()
        decoder.train()
        
        ONES = Variable(Tensor(image_batch.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
        ZEROS = Variable(Tensor(image_batch.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
        
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss

        g_loss = 0.01 * disc_loss_fn(discriminator(encoded_data), ONES) + 0.99 * reconstruction_loss_fn(decoded_data, image_batch)
        # Backward pass
        optim.zero_grad()
        g_loss.backward()
        optim.step()
        # Print batch loss

        train_loss_g.append(g_loss.detach().cpu().numpy())

        encoder.eval()
        decoder.eval()
        optim_d.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (image_batch.shape[0], 8)))).to(device)

        # Discriminator Loss
        real_loss = disc_loss_fn(discriminator(z), ONES)
        fake_loss = disc_loss_fn(discriminator(encoded_data.detach()), ZEROS)
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optim_d.step()
        train_loss_d.append(d_loss.item())
    
    time_taken = time.time() - start_time

    gen_loss = np.mean(train_loss_g)
    disc_loss = np.mean(train_loss_d)
    print(f"Epoch {i+1}/{num_epochs} Generator Loss: {gen_loss} Discriminator Loss: {disc_loss} Time: {time_taken}")
    gen_losses.append(gen_loss)
    disc_losses.append(disc_loss)
    training_times.append(time_taken)


states = {
    "encoder_state": encoder.state_dict(),
    "decoder_state": decoder.state_dict(),
    "discriminator_state": discriminator.state_dict(),
    "ae_loss": gen_losses,
    "discriminator_loss": disc_losses,
    "training_time": training_times,
    "optimizer_ae": optim.state_dict(),
    "optimizer_discriminator": optim_d.state_dict(),
    "num_parameters": num_parameters
}

torch.save(states, "adversarial_ae.pt")