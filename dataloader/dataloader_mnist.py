import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor



def training_dataset(**kwargs):
    return datasets.FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform=ToTensor(),
            )

def test_dataset(**kwargs):
    return datasets.FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform=ToTensor(),
            )
