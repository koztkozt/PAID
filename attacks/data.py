import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class perturbed_data(Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        img = self.x_train[idx]
        angle = self.y_train["angle_new"].iloc[idx]
        angle_gps = self.y_train["angle_gps"].iloc[idx]
        Anomaly = self.y_train["Anomaly"].iloc[idx]
        return img, angle, angle_gps, Anomaly

