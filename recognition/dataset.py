# dataset.py
import os, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_transforms(img_size=224, gray=True, aug=True):
    raise NotImplementedError("transforms not implemented yet")

def get_loaders(data_root="ADNI/AD_NC", img_size=224, batch_size=32, num_workers=4, gray=True):
    raise NotImplementedError("loaders not implemented yet")
