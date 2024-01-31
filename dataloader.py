import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def load_data():
    transform = transforms.ToTensor()

    train_dataset = MNIST(root='data', train=True, transform=transform, download=True)
    val_dataset = MNIST(root='data', train=False, transform=transform, download=True)

    return train_dataset, val_dataset


def dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
