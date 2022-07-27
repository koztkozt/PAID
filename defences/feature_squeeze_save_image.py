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

    df = pd.DataFrame(columns=["output", "output_bit","output_blur","output_adv", "output_adv_bit","output_adv_blur"])
    inter = 0.0
    for i, sample in enumerate(test_data_loader):
        if i != 2560:
            continue
            
        if i / len(test_data_loader) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1

        image = sample[0].type(torch.FloatTensor)
        image = image.to(device)
        
        squeeze_image_bit = reduce_bit(sample[0], 4)
        squeeze_image_bit = squeeze_image_bit.type(torch.FloatTensor)
        squeeze_image_bit = squeeze_image_bit.to(device)
        pred_bit = stage1(squeeze_image_bit)

        squeeze_image_blur = median_filter_np(np.transpose(sample[0].squeeze(0).numpy(), (1, 2, 0)), 2)
        squeeze_image_blur = torch.from_numpy(np.transpose(squeeze_image_blur, (-1, 0, 1))).unsqueeze(0)
        squeeze_image_blur = squeeze_image_blur.to(device)
        pred_blur = stage1(squeeze_image_blur)
        
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
        
        

        
        # print (pred,pred_squeezed)
        output, output_bit, output_blur = stage2_pred(device, stage2, pred, pred_bit, pred_blur, sample)
        # score = torch.sum(torch.abs(2 * (output - output_squeeze)), 1)

        # print (pred_adv,pred_adv_squeezed)
        output_adv, output_adv_bit, output_adv_blur = stage2_pred(device, stage2, pred_adv, pred_adv_bit, pred_adv_blur, sample)
        # score_adv = torch.sum(torch.abs(2 * (output_adv - output_squeeze_adv)), 1)
    
        df.loc[len(df.index)] = [output, output_bit, output_blur,output_adv, output_adv_bit, output_adv_blur]
        
        # if i % 512 == 0:
        #     Path("results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/").mkdir(
        #         parents=True, exist_ok=True
        #     )
        #     plt_ = generate_image(image, squeeze_image_bit, squeeze_image_blur, perturbed_image, squeeze_perturbed_image_bit,squeeze_perturbed_image_blur)
        #     plt_.savefig(
        #         "results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/" + str(i) + ".jpg",
        #         bbox_inches="tight",
        #     )
        #     plt_.close() 
        
        np.save('report_image/perturbed_image_' + attack_name, perturbed_image)
        np.save('report_image/squeeze_perturbed_image_blur_' + attack_name, squeeze_perturbed_image_blur.detach().cpu().numpy())
        np.save('report_image/squeeze_perturbed_image_bit_' + attack_name, squeeze_perturbed_image_bit.detach().cpu().numpy()) 
        
    return df


def defences(config):
    dataset_name = config.dataset_name
    dirname = os.path.dirname(os.path.abspath(__file__))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = config.data_path
    batch_size = config.batch_size

    stage1_path = os.path.join(dirparent, "stage1/stage1_" + sys.argv[3] +"_"+ dataset_name + ".pt")
    if sys.argv[2] == "none":
        stage2_path = os.path.join(dirparent, "stage2/stage2_" + sys.argv[3] +"_" + dataset_name + "_" + "abrupt" + ".pt")
    else:
        stage2_path = os.path.join(dirparent, "stage2/stage2_" + sys.argv[3] +"_" + dataset_name + "_" + sys.argv[2] + ".pt")

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
    Y = pd.read_csv(dirparent + "/" + data_path + "/Y_all_attack_" + sys.argv[2] + ".csv")
    # full_dataset = stage2_data(X, Y)
    X_train, X_rem, Y_train, Y_rem = train_test_split(X, Y, test_size=0.3, random_state=56)
    X_valid, X_test, Y_valid, Y_test  = train_test_split(X_rem, Y_rem, test_size=0.5, random_state=56)
    # train_dataset = stage2_data(X_train, Y_train)
    test_dataset = stage2_data(X_test, Y_test)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    num_sample = len(test_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attack_names = ["fgsm", "opt", "OptU", "advGAN", "advGANU"]
    # attack_names = ["advGANU"]
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
        df["score_bit"] = abs(2 * (df["output"] - df["output_bit"]))
        df["score_blur"] = abs(2 * (df["output"] - df["output_blur"]))
        df["max_score"] = df[["score_bit","score_blur"]].max(axis=1)
        
        df["score_adv_bit"] = abs(2 * (df["output_adv"] - df["output_adv_bit"]))
        df["score_adv_blur"] = abs(2 * (df["output_adv"] - df["output_adv_blur"]))
        df["max_score_adv"] = df[["score_adv_bit","score_adv_blur"]].max(axis=1)
        # df.to_csv("./results/score_" + dataset_name + "_" + sys.argv[2] + "_" + attack_names[i] + ".csv")
        
        df_detection = pd.DataFrame(columns=["orig_detected_bit", "adv_detected_bit", "orig_detected_blur", "adv_detected_blur","orig_detected_both", "adv_detected_both"])
        
        for threshold in [0.01, 0.25, 0.5, 0.75, 1.0, 1.5, 1.99]:
            df_detection.loc[threshold, "orig_detected_bit"] = (df["score_bit"] > threshold).sum()
            df_detection.loc[threshold, "orig_detected_blur"] = (df["score_blur"] > threshold).sum()
            df_detection.loc[threshold, "orig_detected_both"] = (df["max_score"] > threshold).sum()
  
            df_detection.loc[threshold, "adv_detected_bit"] = (df["score_adv_bit"] > threshold).sum()
            df_detection.loc[threshold, "adv_detected_blur"] = (df["score_adv_blur"] > threshold).sum()
            df_detection.loc[threshold, "adv_detected_both"] = (df["max_score_adv"] > threshold).sum()
        
        for detected in ["orig_detected_bit", "adv_detected_bit", "orig_detected_blur", "adv_detected_blur","orig_detected_both", "adv_detected_both"]:
             df_detection[detected+"_rate"] =  df_detection[detected] / num_sample

        print("Detection results")
        print(df_detection)
        # df_detection.to_csv("./results/" + dataset_name + "_" + sys.argv[2] + "_" + attack_names[i] + ".csv")

def generate_image(image, squeeze_image_bit, squeeze_image_blur, perturbed_image, squeeze_perturbed_image_bit,squeeze_perturbed_image_blur):
    ax1 = plt.subplot(2, 3, 1)
    ax1.title.set_text("original image")
    plt.imshow(image.detach().cpu().numpy()[0, 0, :, :], cmap="gray")
    ax2 = plt.subplot(2, 3, 2)
    ax2.title.set_text("bit reduction")
    plt.imshow(squeeze_image_bit.detach().cpu().numpy()[0, 0, :, :], cmap="gray")
    ax3 = plt.subplot(2, 3, 3)
    ax3.title.set_text("median filter")
    plt.imshow(squeeze_image_blur.detach().cpu().numpy()[0, 0, :, :], cmap="gray")
    ax4 = plt.subplot(2, 3, 4)
    ax4.title.set_text("perturbed image")
    plt.imshow(perturbed_image[0, 0, :, :], cmap="gray")
    ax5 = plt.subplot(2, 3, 5)
    ax5.title.set_text("bit reduction")
    plt.imshow(squeeze_perturbed_image_bit.detach().cpu().numpy()[0, 0, :, :], cmap="gray")
    ax6 = plt.subplot(2, 3, 6)
    ax6.title.set_text("median filter")
    plt.imshow(squeeze_perturbed_image_blur.detach().cpu().numpy()[0, 0, :, :], cmap="gray")
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)

    return plt

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


if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.defencesconfig()
    defences(config)
