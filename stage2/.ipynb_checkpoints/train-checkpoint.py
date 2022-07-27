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
from torchinfo import summary
from torch.utils.data import DataLoader
import cv2
from collections import deque
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from train import *
from stage1.model import stage1
from model import stage2
from data import stage2_data

if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.TrainConfig2()
    dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    ch = config.num_channels
    num_epoch = config.num_epoch
    batch_size = config.batch_size
    data_path = config.data_path
    dataset_name = config.dataset_name
    target_model_path = os.path.join(dirname, "stage1/stage1_" + sys.argv[3] + "_" + dataset_name + ".pt")

    print("Loading training data...")
    if sys.argv[2] == "evade":
        X = np.load(dirparent + "/" + data_path +"X_all.npy")
        X = X[70:-70]       
        Y = pd.read_csv(dirparent + "/" + data_path + "Y_all_attack_" + sys.argv[2] + ".csv")
        X_train, X_rem, Y_train, Y_rem = train_test_split(X,Y, test_size=0.3, random_state=6)
        X_valid, X_test, Y_valid, Y_test = train_test_split(X_rem,Y_rem, test_size=0.5, random_state=6)
    else:
        X_train = np.load(dirparent + "/" + data_path + "X_train.npy")
        Y_train = pd.read_csv(dirparent + "/" + data_path + "Y_train_attack_" + sys.argv[2] + ".csv")
        X_valid = np.load(dirparent + "/" + data_path + "X_valid.npy")
        Y_valid = pd.read_csv(dirparent + "/" + data_path + "Y_valid_attack_" + sys.argv[2] + ".csv")

    print("Creating model...")
    train_dataset = stage2_data(X_train, Y_train)
    train_generator = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    train_steps_per_epoch = int(len(train_dataset) / batch_size)
    test_dataset = stage2_data(X_valid, Y_valid)
    test_generator = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    test_steps_per_epoch = int(len(test_dataset) / batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load stage1 model
    stage1_model = stage1()
    stage1_model= stage1_model.to(device)
    stage1_model.load_state_dict(torch.load(target_model_path))
    stage1_model.eval()

    net = stage2()
    net = net.to(device)
    summary(net, input_size=(batch_size, 1))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    min_valid_loss = 1000000

    print("Length of dataloader :", len(train_generator))
    df_all = pd.DataFrame(columns=["TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall"])
    for epoch in range(num_epoch):
        net.train(True)
        running_loss = 0
        inter = 0.0
        for i_batch, sample in enumerate(train_generator):  # for each training i_batch
            if i_batch / len(train_generator) > inter:
                print(f"epoch: {epoch+1} completed: {(inter):.0%}")
                inter += 0.10

            batch_x, angle, target = sample
            batch_x = batch_x.type(torch.FloatTensor)
            angle = angle.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            angle = angle.to(device)
            target = target.to(device)

            predicted_angle = stage1_model(batch_x)

            final_vars = torch.abs(torch.sub(angle.unsqueeze(-1), predicted_angle))

            output = net(final_vars)
            # values, indices = torch.max(output, dim=1)
            loss = criterion(output, target.unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # pred = np.round(torch.sigmoid(output.detach()))
            # print(pred.reshape(-1))
            # print(target)
            # break
        avg_loss = running_loss / train_steps_per_epoch
        # print(f"Epoch {epoch+1} loss: {(avg_loss):4f}")

        print("########################### VALIDATION ##########################")
        net.train(False)
        running_vloss = 0.0
        y_pred = []
        y_true = []
        for _, sample in enumerate(test_generator):
            batch_x, angle, target = sample
            batch_x = batch_x.type(torch.FloatTensor)
            angle = angle.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            angle = angle.to(device)
            target = target.to(device)

            predicted_angle = stage1_model(batch_x)

            final_vars = torch.abs(torch.sub(angle.unsqueeze(-1), predicted_angle))

            output = net(final_vars)
            loss = criterion(output, target.unsqueeze(-1))

            running_vloss += loss.item()

            pred = np.round(torch.sigmoid(output.detach().cpu()))
            # print(pred.reshape(-1))
            # print(target)
            y_pred.extend(pred.reshape(-1))
            y_true.extend(target.detach().cpu().data)

        avg_vloss = running_vloss / test_steps_per_epoch
        print(f'Epoch {epoch+1} \t\t Training Loss: {avg_loss} \t\t Validation Loss: {avg_vloss}')   
        if avg_vloss < min_valid_loss:
            min_valid_loss = avg_vloss
            torch.save(net.state_dict(), "stage2_" + sys.argv[3] + "_" + dataset_name + "_" + sys.argv[2] + ".pt")

            # Build confusion matrix
            cf_matrix = confusion_matrix(y_true, y_pred).flatten()
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            print("TN, FP, FN, TP :", cf_matrix)
            print("accuracy :", accuracy)

            # Initialize a blank dataframe and keep adding
            df = pd.DataFrame(columns=["TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall"])
            df.loc[epoch] = cf_matrix.tolist() + [accuracy, precision, recall]
            df["Total_Actual_Neg"] = df["TN"] + df["FP"]
            df["Total_Actual_Pos"] = df["FN"] + df["TP"]
            df["Total_Pred_Neg"] = df["TN"] + df["FN"]
            df["Total_Pred_Pos"] = df["FP"] + df["TP"]
            df["TP_Rate"] = df["TP"] / df["Total_Actual_Pos"]  # Recall
            df["FP_Rate"] = df["FP"] / df["Total_Actual_Neg"]
            df["TN_Rate"] = df["TN"] / df["Total_Actual_Neg"]
            df["FN_Rate"] = df["FN"] / df["Total_Actual_Pos"]
            df_all = pd.concat([df_all, df])
            print(df_all.tail())

    df_all.to_csv("accurracy_" +  sys.argv[3] + "_" + dataset_name + "_" + sys.argv[2] + ".csv")
