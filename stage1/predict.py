from sklearn.model_selection import train_test_split
import os, sys, importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import math
import csv
from os import path
import numpy as np
import pandas as pd
import time
from torchvision import transforms, utils
from torchinfo import summary
from torch.utils.data import DataLoader
import cv2
from model import stage1
from data import stage1_data
from sklearn.metrics import mean_squared_error 

matplotlib.use("Agg")

if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.TrainConfig1()
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    ch = config.num_channels
    num_epoch = config.num_epoch
    batch_size = config.batch_size
    data_path = config.data_path
    dataset_name = config.dataset_name
    img_height = config.img_height
    img_width = config.img_width
    num_channels = config.num_channels
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_path = os.path.join("stage1_" + sys.argv[2] + "_" + dataset_name + ".pt")
    stage1_model = stage1().to(device)
    stage1_model.load_state_dict(torch.load(stage1_path))
    stage1_model.eval()
    
    print("Loading testing data...")
    X_test = np.load(dirparent + "/" + data_path + "X_test.npy")
    Y_test = pd.read_csv(dirparent + "/" + data_path + "Y_test_attack_none.csv")

    print("Creating model...")
    test_dataset = stage1_data(X_test, Y_test)
    test_generator = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    test_steps_per_epoch = int(len(test_dataset) / batch_size)
    
    criterion = nn.MSELoss()
    running_vloss = 0.0
    yhat = []
    test_y = []
    for i, sample in enumerate(test_generator):
        if sys.argv[2] == "angle":
            batch_x, batch_y = sample[0], sample[1]
        elif sys.argv[2] == "speed":    
            batch_x, batch_y = sample[0], sample[2]

        batch_x = batch_x.type(torch.FloatTensor)
        batch_y = batch_y.type(torch.FloatTensor)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = stage1_model(batch_x)
        loss = criterion(outputs, batch_y.unsqueeze(-1))
        running_vloss += loss.item()

        yhat.append(outputs.tolist())
        test_y.append(batch_y.tolist())

    avg_vloss = running_vloss / test_steps_per_epoch
    print(f'Testing Loss: {avg_vloss}')   

    min_valid_loss = avg_vloss

    yhat = np.concatenate(yhat).ravel()
    test_y = np.concatenate(test_y).ravel()
    # rmse = np.sqrt(np.mean((yhat - test_y) ** 2)) / (max(test_y) - min(test_y))
    mse = mean_squared_error(test_y, yhat)
    print("mse: ", mse)
    plt.figure(figsize=(32, 8))
    plt.plot(test_y, "r.-", label="target")
    plt.plot(yhat, "b^-", label="predict")
    plt.legend(loc="best")
    plt.title("MSE: %.4f" % mse)
    plt.show()
    model_fullname = "test_%s_%s_%d.png" % (dataset_name, sys.argv[2], int(time.time()))
    plt.savefig(model_fullname)
