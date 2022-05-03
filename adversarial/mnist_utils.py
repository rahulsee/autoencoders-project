import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split

class MNIST_Util:
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
        self.test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)
        self.train_dataset.transform = transforms.Compose([transforms.ToTensor()])
        self.test_dataset.transform = transforms.Compose([transforms.ToTensor()])
        self.batch_size=batch_size

    def get_train_loader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)
        
    def get_test_loader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)