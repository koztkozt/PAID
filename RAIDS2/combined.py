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
from torch.utils.data import Dataset, DataLoader
import cv2
from collections import deque
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from attacks.advGAN.models import Generator
from attacks.advGAN_attack import advGAN_Attack
from attacks.attacking import advGAN_ex, advGANU_ex, fgsm_ex, opt_ex, opt_uni_ex
from attacks.fgsm_attack import fgsm_attack
from attacks.optimization_attack import optimized_attack
from attacks.optimization_universal_attack import generate_noise
from attacks.attack_test import (
    fgsm_attack_test,
    optimized_attack_test,
    optimized_uni_test,
    advGAN_test,
    advGAN_uni_test,
)
from scipy import ndimage
from stage1.model import stage1
from stage1a.model import stage1a
from stage2.model import stage2
from stage2a.model import stage2a
from data import combined_data

def reduce_bit(image, bit_size):
    image_int = np.rint(image * (math.pow(2, bit_size) - 1))
    image_float = image_int / (math.pow(2, bit_size) - 1)
    return image_float


def median_filter_np(x, width, height=-1):
    """
    Median smoothing by Scipy.
    :param x: a tensor of image(s)
    :param width: the width of the sliding window (number of pixels)
    :param height: the height of the window. The same as width by default.
    :return: a modified tensor with the same shape as x.
    """
    if height == -1:
        height = width
    return ndimage.median_filter(x, size=2, mode="reflect")

def stage2_pred(device, stage2, pred, pred_bit, pred_blur, sample):
    batch_x = sample[0]
    angle = sample[1]
    target = sample[3]
    batch_x = batch_x.type(torch.FloatTensor)
    angle = angle.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)
    batch_x = batch_x.to(device)
    angle = angle.to(device)
    target = target.to(device)

    output = stage2(torch.abs(torch.sub(angle.unsqueeze(-1), pred)))
    output_bit = stage2(torch.abs(torch.sub(angle.unsqueeze(-1), pred_bit)))
    output_blur = stage2(torch.abs(torch.sub(angle.unsqueeze(-1), pred_blur)))
    
    return torch.sigmoid(output.detach()).item(), torch.sigmoid(output_bit.detach()).item(),torch.sigmoid(output_blur.detach()).item()

def feature_squeeze(stage1, stage2, config, sample, image, pred, attack_name):

    squeeze_image_bit = reduce_bit(image, 4)
    squeeze_image_bit = squeeze_image_bit.type(torch.FloatTensor)
    squeeze_image_bit = squeeze_image_bit.to(device)
    pred_bit = stage1(squeeze_image_bit)

    squeeze_image_blur = median_filter_np(np.transpose(image.squeeze(0).numpy(), (1, 2, 0)), 2)
    squeeze_image_blur = torch.from_numpy(np.transpose(squeeze_image_blur, (-1, 0, 1))).unsqueeze(0)
    squeeze_image_blur = squeeze_image_blur.to(device)
    pred_blur = stage1(squeeze_image_blur)
    # print (pred,pred_squeezed)
    output, output_bit, output_blur = stage2_pred(device, stage2, pred, pred_bit, pred_blur, sample)

    score_bit = abs(2 * (output - output_bit)) 
    score_blur = abs(2 * (output - output_blur))
    score_both = max(score_bit, score_blur)

    threshold = {
      "fgsm": 0.01,
      "opt": 0.12, 
      "OptU": 0.05, 
      "advGAN": 0.52,        
      "advGANU": 0.12} 

    if attack_name == "fgsm":
        score = score_both
    elif attack_name == "opt":  
        score = score_blur
    elif attack_name == "OptU":        
        score = score_blur
    elif attack_name == "advGAN":
        score = score_both
    elif attack_name == "advGANU":
        score = score_both

    # if score > 0.3:
    # print(score)
    if score > threshold[attack_name]:
        pred = 1
    else:
        pred = 0
    
    return pred

if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.combined()
    dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ch = config.num_channels
    batch_size = config.batch_size
    data_path = config.data_path
    dataset_name = config.dataset_name
    target_angle = config.target
    image_nc = config.num_channels
    image_size = (config.img_height, config.img_width)
    attack_name = sys.argv[4]
    
    stage1_path = os.path.join(dirname, "stage1/stage1_" + sys.argv[3] + "_" + dataset_name + ".pt")
    if sys.argv[2] == "none":
        stage2_path = os.path.join(dirparent, "stage2/stage2_" + sys.argv[3] + "_" + dataset_name + "_" + "abrupt" + ".pt")
    else:
        stage2_path = os.path.join(dirparent, "stage2/stage2_" + sys.argv[3] + "_" + dataset_name + "_" + sys.argv[2] + ".pt")
    stage2a_path = os.path.join(dirname, "stage2a/stage2a_" + sys.argv[3] +"_"+ dataset_name + ".pt")
    
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        
    stage1_model = stage1().to(device)
    stage2_model = stage2().to(device)
    stage1a_model = stage1a().to(device)
    stage2a_model = stage2a().to(device)
    
    if torch.cuda.is_available():
        stage1_model.load_state_dict(torch.load(stage1_path))
        stage2_model.load_state_dict(torch.load(stage2_path))
        stage2a_model.load_state_dict(torch.load(stage2a_path))
    else:
        stage1_model.load_state_dict(torch.load(stage1_path,map_location=torch.device('cpu')))
        stage2_model.load_state_dict(torch.load(stage2_path,map_location=torch.device('cpu')))
        stage2a_model.load_state_dict(torch.load(stage2a_path,map_location=torch.device('cpu')))

    stage1_model.eval()
    stage2_model.eval()
    stage1a_model.eval()
    stage2a_model.eval()
    
    print("Loading testing data...")
    X = np.load(dirparent + "/" + data_path + "X_all.npy")
    X = X[70:-70]
    Y = pd.read_csv(dirparent + "/" + data_path + "/Y_all_attack_evade.csv")
    X_train, X_rem, Y_train, Y_rem = train_test_split(X, Y, test_size=0.3, random_state=56)
    X_valid, X_test, Y_valid, Y_test  = train_test_split(X_rem, Y_rem, test_size=0.5, random_state=56)
    
    test_dataset = combined_data(X_test, Y_test)
    test_generator = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    test_steps_per_epoch = int(len(test_dataset) / batch_size)
    
    criterion = nn.BCEWithLogitsLoss()
    running_vloss = 0.0
    y_pred = []
    y_true = []
    inter = 0.0
    print("attacking RAIDS using ", attack_name) 
    print("Length of dataloader :", len(test_generator))
    for i, sample in enumerate(test_generator):
        if i / len(test_generator) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1
        img, angle, angle_gps, target = sample
        # print(angle, angle_gps, stage1_angle, target)
        img = img.type(torch.FloatTensor)
        angle = angle.type(torch.FloatTensor)
        angle_gps = angle_gps.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)
        
        img = img.to(device)
        angle = angle.to(device)
        angle_gps = angle_gps.to(device)
        target = target.to(device)

        # stage1
        if not target:
            # print("not attack")
            predicted_angle = stage1_model(img)
            image = sample[0]
        else:
            # print("attack")
            plot_fig = False
            if attack_name == "fgsm":
                diff, pred, pred_adv, plt_, _, perturbed_image = fgsm_attack_test(
                    stage1_model, sample[0], target_angle, device, image_size, plot_fig
                )
            elif attack_name == "opt":
                diff, pred, pred_adv, plt_, _, perturbed_image = optimized_attack_test(
                    stage1_model, sample[0], target_angle, device, image_size, plot_fig
                )
            elif attack_name == "OptU":
                noise = np.load(dirparent +"/attacks/models/" + dataset_name + "_" + attack_name + "_noise_seed.npy")
                noise = np.tile(noise, (batch_size, 1, 1, 1))
                noise = torch.from_numpy(noise).type(torch.FloatTensor).to(device)
                diff, pred, pred_adv, plt_, perturbed_image = optimized_uni_test(
                    stage1_model, sample[0], device, noise, image_size, plot_fig
                )
                
            elif attack_name == "advGAN":
                advGAN_generator = Generator(image_nc, image_nc, attack_name).to(device)
                if torch.cuda.is_available():
                    advGAN_generator.load_state_dict(torch.load(dirparent +"/attacks/models/" + dataset_name + "_" + attack_name + "_netG_epoch_60.pth"))
                else:
                    advGAN_generator.load_state_dict(torch.load(dirparent +"/attacks/models/" + dataset_name + "_" + attack_name + "_netG_epoch_60.pth",map_location=torch.device('cpu')))
                advGAN_generator.eval()
                # print(steer, adv_output)
                diff, pred, pred_adv, plt_,  _, perturbed_image = advGAN_test(
                    stage1_model, sample[0], advGAN_generator, device, image_size, plot_fig
                )
            elif attack_name == "advGANU":
                advGANU_generator = Generator(image_nc, image_nc, attack_name).to(device)
                if torch.cuda.is_available():
                    advGANU_generator.load_state_dict(torch.load(dirparent +"/attacks/models/" + dataset_name + "_" + attack_name + "_netG_epoch_60.pth"))
                else:
                    advGANU_generator.load_state_dict(torch.load(dirparent +"/attacks/models/" + dataset_name + "_" + attack_name + "_netG_epoch_60.pth",map_location=torch.device('cpu')))
                advGANU_generator.eval()
                noise_seed = np.load(dirparent + "/attacks/models/" + dataset_name + "_" + attack_name + "_noise_seed.npy")
                noise_seed = np.tile(noise_seed, (batch_size, 1, 1, 1))
                noise = advGANU_generator(torch.from_numpy(noise_seed).type(torch.FloatTensor).to(device))
                diff, pred, pred_adv, plt_, perturbed_image = advGAN_uni_test(
                    stage1_model, sample[0], device, noise, image_size, plot_fig
                )
            
            predicted_angle = pred_adv
            image = torch.from_numpy(perturbed_image)
        
        if sys.argv[5] == "RAIDS2":
            # stage1a
            stage1a_intrusion = feature_squeeze(stage1_model, stage2_model, config, sample, image, predicted_angle, attack_name)
            if stage1a_intrusion:
                # use GPS
                final_vars = torch.abs(torch.sub(angle.unsqueeze(-1), angle_gps.unsqueeze(-1)))
            else:
                # use image
                final_vars = torch.abs(torch.sub(angle.unsqueeze(-1), predicted_angle))
            output = stage2_model(final_vars)
        elif sys.argv[5] == "RAIDS":
            final_vars = torch.abs(torch.sub(angle.unsqueeze(-1), predicted_angle))
            output = stage2_model(final_vars)
        
        loss = criterion(output, target.unsqueeze(-1))
        running_vloss += loss.item()

        prediction = np.round(torch.sigmoid(output.detach().cpu()))

        y_pred.extend(prediction.reshape(-1))
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
    df.to_csv("accurracy_" + sys.argv[3] + "_" + dataset_name + "_" + sys.argv[4] + "_" + sys.argv[5] + ".csv")
