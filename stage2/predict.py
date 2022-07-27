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
    
    stage1_path = os.path.join(dirname, "stage1/stage1_" + sys.argv[3] + "_" + dataset_name + ".pt")
    if sys.argv[2] == "none":
        stage2_path = os.path.join(dirparent, "stage2/stage2_" + sys.argv[3] + "_" + dataset_name + "_" + "abrupt" + ".pt")
    else:
        stage2_path = os.path.join(dirparent, "stage2/stage2_" + sys.argv[3] + "_" + dataset_name + "_" + sys.argv[2] + ".pt")
    
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    
    stage1_model = stage1().to(device)
    stage1_model.load_state_dict(torch.load(stage1_path))
    stage1_model.eval()
    stage2_model = stage2().to(device)
    stage2_model.load_state_dict(torch.load(stage2_path))
    stage2_model.eval()
    
    print("Loading testing data...")
    if sys.argv[2] == "evade":
        X = np.load(dirparent + "/" + data_path +"X_all.npy")
        X = X[70:-70]       
        Y = pd.read_csv(dirparent + "/" + data_path + "Y_all_attack_" + sys.argv[2] + ".csv")
        Y = Y[Y['angle_gps'].notna()]
        X_train, X_rem, Y_train, Y_rem = train_test_split(X,Y, test_size=0.3, random_state=6)
        X_valid, X_test, Y_valid, Y_test = train_test_split(X_rem,Y_rem, test_size=0.5, random_state=6)
    else:
        X_test = np.load(dirparent + "/" + data_path + "X_test.npy")
        Y_test = pd.read_csv(dirparent + "/" + data_path + "Y_test_attack_" + sys.argv[2] + ".csv")
        
    test_dataset = stage2_data(X_test, Y_test)
    test_generator = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    test_steps_per_epoch = int(len(test_dataset) / batch_size)
    
    criterion = nn.BCEWithLogitsLoss()
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
        output = stage2_model(final_vars)
        
        loss = criterion(output, target.unsqueeze(-1))

        running_vloss += loss.item()

        pred = np.round(torch.sigmoid(output.detach().cpu()))
        # print(pred.reshape(-1))
        # print(target)
        y_pred.extend(pred.reshape(-1))
        y_true.extend(target.detach().cpu().data)
    
    avg_vloss = running_vloss / test_steps_per_epoch
    print(f'Testing Loss: {avg_vloss}')    
    
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred).flatten()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print("TN, FP, FN, TP :", cf_matrix)
    print("accuracy :", accuracy)

    # Initialize a blank dataframe and keep adding
    df = pd.DataFrame(columns=["TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall"])
    df.loc[1] = cf_matrix.tolist() + [accuracy, precision, recall]
    df["Total_Actual_Neg"] = df["TN"] + df["FP"]
    df["Total_Actual_Pos"] = df["FN"] + df["TP"]
    df["Total_Pred_Neg"] = df["TN"] + df["FN"]
    df["Total_Pred_Pos"] = df["FP"] + df["TP"]
    df["TP_Rate"] = df["TP"] / df["Total_Actual_Pos"]  # Recall
    df["FP_Rate"] = df["FP"] / df["Total_Actual_Neg"]
    df["TN_Rate"] = df["TN"] / df["Total_Actual_Neg"]
    df["FN_Rate"] = df["FN"] / df["Total_Actual_Pos"]        
    print(df.tail())
    df.to_csv("accurracy_" +  sys.argv[3] + "_" + dataset_name + "_" + sys.argv[2] + ".csv")
