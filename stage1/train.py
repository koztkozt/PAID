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

    print("Loading training data...")
    X_train = np.load(dirparent + "/" + data_path + "X_train.npy")
    Y_train = pd.read_csv(dirparent + "/" + data_path + "Y_train_attack_none.csv")
    X_valid = np.load(dirparent + "/" + data_path + "X_valid.npy")
    Y_valid = pd.read_csv(dirparent + "/" + data_path + "Y_valid_attack_none.csv")

    print("Creating model...")
    train_dataset = stage1_data(X_train, Y_train)
    train_generator = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    train_steps_per_epoch = int(len(train_dataset) / batch_size)
    test_dataset = stage1_data(X_valid, Y_valid)
    test_generator = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    test_steps_per_epoch = int(len(test_dataset) / batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = stage1()
    net = net.to(device)
    summary(net, input_size=(batch_size, num_channels, img_height, img_width))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    min_valid_loss = 1000000
    for epoch in range(num_epoch):
        net.train(True)
        running_loss = 0
        inter = 0.0
        for i_batch, sample in enumerate(train_generator):  # for each training i_batch
            if i_batch / len(train_generator) > inter:
                print(f"epoch: {epoch+1} completed: {(inter):.0%}")
                inter += 0.1
            if sys.argv[2] == "angle":
                batch_x, batch_y = sample[0], sample[1]
            elif sys.argv[2] == "speed":    
                batch_x, batch_y = sample[0], sample[2]
            batch_x = batch_x.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = net(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / train_steps_per_epoch
        # print(f"Epoch {epoch} RMSE loss: {(avg_loss):4f}")

        print("########################### VALIDATION ##########################")
        net.train(False)
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

            outputs = net(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(-1))
            running_vloss += loss.item()

            yhat.append(outputs.tolist())
            test_y.append(batch_y.tolist())

        avg_vloss = running_vloss / test_steps_per_epoch
        print(f'Epoch {epoch+1} \t\t Training Loss: {avg_loss} \t\t Validation Loss: {avg_vloss}')   
        
        if avg_vloss < min_valid_loss:
            min_valid_loss = avg_vloss
            torch.save(net.state_dict(), "stage1_" + sys.argv[2] +"_"+ dataset_name + ".pt")
            
            # yhat = np.concatenate(yhat).ravel()
            # test_y = np.concatenate(test_y).ravel()
            # rmse = np.sqrt(np.mean((yhat - test_y) ** 2)) / (max(test_y) - min(test_y))
            # plt.figure(figsize=(32, 8))
            # plt.plot(test_y, "r.-", label="target")
            # plt.plot(yhat, "b^-", label="predict")
            # plt.legend(loc="best")
            # plt.title("RMSE: %.2f" % rmse)
            # plt.show()
            # model_fullname = "%s_%s_%d.png" % (dataset_name, sys.argv[2], int(time.time()))
            # plt.savefig(model_fullname)
