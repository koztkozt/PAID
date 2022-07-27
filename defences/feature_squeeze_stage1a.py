from sklearn.model_selection import train_test_split
import importlib
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import argparse
import csv
import math
import time

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# from scipy.misc import imread, imresize, imsave

from stage1.model import stage1
from stage2.model import stage2
from stage2.data import stage2_data


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


def attack_detection(
    stage1,
    stage2,
    test_data_loader,
    num_sample,
    config,
    dirparent,
    attack_name,
    device,
    thresholds=[0.01, 0.5, 1.0, 1.5, 1.99],
):
    dataset_name = config.dataset_name
    batch_size = config.batch_size
    target = config.target
    image_nc = config.num_channels
    image_size = (config.img_height, config.img_width)
    threshold = config.threshold
    
    y_pred = []
    y_true = []
    inter = 0.0
    df = pd.DataFrame(columns=["output", "output_bit","output_blur","score_bit","score_blur","anomaly"])
    for i, sample in enumerate(test_data_loader):
        if i / len(test_data_loader) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1

        image = sample[0].type(torch.FloatTensor)
        image = image.to(device)
                
        attack = sample[2]
        if not attack:
            pred = stage1(image)
            squeeze_image_bit = reduce_bit(sample[0], 4)
            squeeze_image_bit = squeeze_image_bit.type(torch.FloatTensor)
            squeeze_image_bit = squeeze_image_bit.to(device)
            pred_bit = stage1(squeeze_image_bit)

            squeeze_image_blur = median_filter_np(np.transpose(sample[0].squeeze(0).numpy(), (1, 2, 0)), 2)
            squeeze_image_blur = torch.from_numpy(np.transpose(squeeze_image_blur, (-1, 0, 1))).unsqueeze(0)
            squeeze_image_blur = squeeze_image_blur.to(device)
            pred_blur = stage1(squeeze_image_blur)
            # print (pred,pred_squeezed)
            output, output_bit, output_blur = stage2_pred(device, stage2, pred, pred_bit, pred_blur, sample)
                      
            
        else:
            plot_fig = False
            if attack_name == "fgsm":
                diff, pred, pred_adv, plt_, _, perturbed_image = fgsm_attack_test(
                    stage1, sample[0], target, device, image_size, plot_fig
                )
            elif attack_name == "opt":
                diff, pred, pred_adv, plt_, _, perturbed_image = optimized_attack_test(
                    stage1, sample[0], target, device, image_size, plot_fig
                )
            elif attack_name == "OptU":
                noise = np.load(dirparent +"/attacks/models/" + dataset_name + "_" + attack_name + "_noise_seed.npy")
                noise = np.tile(noise, (batch_size, 1, 1, 1))
                noise = torch.from_numpy(noise).type(torch.FloatTensor).to(device)
                diff, pred, pred_adv, plt_, perturbed_image = optimized_uni_test(
                    stage1, sample[0], device, noise, image_size, plot_fig
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
                    stage1, sample[0], advGAN_generator, device, image_size, plot_fig
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
                    stage1, sample[0], device, noise, image_size, plot_fig
                )

            squeeze_perturbed_image_bit = reduce_bit(perturbed_image, 4)
            squeeze_perturbed_image_bit = torch.from_numpy(squeeze_perturbed_image_bit)
            squeeze_perturbed_image_bit = squeeze_perturbed_image_bit.to(device)
            pred_adv_bit= stage1(squeeze_perturbed_image_bit)

            squeeze_perturbed_image_blur = median_filter_np(perturbed_image, 2)
            squeeze_perturbed_image_blur = torch.from_numpy(squeeze_perturbed_image_blur)
            squeeze_perturbed_image_blur = squeeze_perturbed_image_blur.to(device)
            pred_adv_blur = stage1(squeeze_perturbed_image_blur)      
    
            output, output_bit, output_blur = stage2_pred(device, stage2, pred_adv, pred_adv_bit, pred_adv_blur, sample)
            # print(output_adv, output_adv_bit)
        score_bit = abs(2 * (output - output_bit)) 
        score_blur = abs(2 * (output - output_blur))
        # print(score)
        if max(score_bit,score_blur) > threshold:
            pred = 1
        else:
            pred = 0
        y_pred.append(pred)
        y_true.append(attack.item())
        # print(pred,attack.item())
        
        df.loc[len(df.index)] = [output, output_bit, output_blur,score_bit, score_blur, attack.item()]
    results(y_true, y_pred,attack_name)
    return df

def defences(config):
    dataset_name = config.dataset_name
    dirname = os.path.dirname(os.path.abspath(__file__))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = config.data_path
    batch_size = config.batch_size

    stage1_path = os.path.join(dirparent, "stage1/stage1_" + sys.argv[3] +"_"+ dataset_name + ".pt")
    stage2_path = os.path.join(dirparent, "stage2/stage2_" + sys.argv[3] +"_" + dataset_name + "_" + "abrupt" + ".pt")

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    stage1_model = stage1().to(device)
    if torch.cuda.is_available():
        stage1_model.load_state_dict(torch.load(stage1_path))
    else:
        stage1_model.load_state_dict(torch.load(stage1_path,map_location=torch.device('cpu')))
    stage1_model.eval()

    stage2_model = stage2().to(device)
    if torch.cuda.is_available():
        stage2_model.load_state_dict(torch.load(stage2_path))
    else:
        stage2_model.load_state_dict(torch.load(stage2_path,map_location=torch.device('cpu')))
    stage2_model.eval()

    # root_dir = "../udacity-data"
    # attacks = ("FGSM", "Optimization", "Optimization Universal", "AdvGAN", "AdvGAN Universal")

    print("Loading testing data...")
    X = np.load(dirparent + "/" + data_path + "X_all.npy")
    X = X[70:-70]
    Y = pd.read_csv(dirparent + "/" + data_path + "/Y_all_attack_evade.csv")
    
    X_train, X_rem, Y_train, Y_rem = train_test_split(X, Y, test_size=0.3, random_state=56)
    X_valid, X_test, Y_valid, Y_test  = train_test_split(X_rem, Y_rem, test_size=0.5, random_state=56)

    test_dataset = stage2_data(X_train, Y_train)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    num_sample = len(test_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    attack_names = ["fgsm", "opt", "OptU", "advGAN", "advGANU"]
    # attack_names = ["advGAN"]
    for i in range(len(attack_names)):
        print("[+] Attacking using", attack_names[i])
        df = attack_detection(
            stage1_model,
            stage2_model,
            test_data_loader,
            num_sample,
            config,
            dirparent,
            attack_names[i],
            device)
        df.to_csv("./results_fs_1a/score_" + dataset_name + "_" + sys.argv[2] + "_" + attack_names[i] + ".csv")

def stage2_pred(device, stage2, pred, pred_bit, pred_blur, sample):
    batch_x, angle, target = sample
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

def results(y_true, y_pred,attack_name):
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
    df.to_csv("./results_fs_1a/accurracy_" +  sys.argv[3] + "_" + "udacity" + "_" + attack_name + ".csv")

if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.defencesconfig()
    defences(config)
