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
from model import stage1, weight_init
from data import stage1_data

matplotlib.use("Agg")

if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.TrainConfig1()

    ch = config.num_channels
    num_epoch = config.num_epoch
    batch_size = config.batch_size
    data_path = config.data_path
    dataset_name = config.dataset_name
    img_height = config.img_height
    img_width = config.img_width
    num_channels = config.num_channels

    print("Loading training data...")
    X = np.load(data_path + "/X_train.npy")
    Y = pd.read_csv(data_path + "/Y_train.csv")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=56)

    # print("Computing training set mean...")
    # X_train_mean = np.mean(X_train, axis=0, keepdims=True)
    # print("Saving training set mean...")
    # np.save(config.X_train_mean_path, X_train_mean)

    print("Creating model...")
    dataset = stage1_data(X_train, Y_train)
    train_generator = DataLoader(dataset, batch_size=batch_size, num_workers=8)
    steps_per_epoch = int(len(dataset) / batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = stage1()
    net = net.to(device)
    net.apply(weight_init)
    summary(net, input_size=(batch_size, num_channels, img_height, img_width))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(num_epoch):
        inter = 0.0
        total_loss = 0
        for i_batch, sample in enumerate(train_generator):  # for each training i_batch
            if i_batch / len(train_generator) > inter:
                print(f"epoch: {epoch} completed: {(inter):.0%}")
                inter += 0.1

            batch_x, batch_y = sample
            batch_x = batch_x.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = net(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            total_loss += running_loss
        print(f"Epoch {epoch} RMSE loss: {(total_loss / steps_per_epoch):4f}")
        torch.save(net.state_dict(), "stage1_" + dataset_name + ".pt")

        print("########################### TESTING ##########################")
        # net.eval()
        # with torch.no_grad():
        yhat = []
        test_y = []
        dataset = stage1_data(X_test, Y_test)
        test_generator = DataLoader(dataset, batch_size=1)
        for _, sample_batched in enumerate(test_generator):
            batch_x, batch_y = sample_batched
            batch_x = batch_x.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = net(batch_x)
            yhat.append(output.item())
            test_y.append(batch_y.item())

        yhat = np.array(yhat)
        test_y = np.array(test_y)
        rmse = np.sqrt(np.mean((yhat - test_y) ** 2)) / (max(test_y) - min(test_y))
        print(rmse)
        plt.figure(figsize=(32, 8))
        plt.plot(test_y, "r.-", label="target")
        plt.plot(yhat, "b^-", label="predict")
        plt.legend(loc="best")
        plt.title("RMSE: %.2f" % rmse)
        plt.show()
        model_fullname = "%s_%d.png" % (dataset_name, int(time.time()))
        plt.savefig(model_fullname)
