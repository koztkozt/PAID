import pandas as pd
import numpy as np
from pathlib import Path
import importlib
import sys
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.attack_test import (
    fgsm_attack_test,
    optimized_attack_test,
    optimized_uni_test,
    advGAN_test,
    advGAN_uni_test,
)
from stage2.data import stage2_data
from stage2.model import stage2
from stage1.model import stage1
from attacks.advGAN_attack import advGAN_Attack
from attacks.optimization_universal_attack import generate_noise
from attacks.optimization_attack import optimized_attack
from attacks.fgsm_attack import fgsm_attack
from attacks.advGAN.models import Generator

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# from viewer import draw

"""
Experiment 1: test the total attack success rate of 5 attacks on 3 models 
"""


def attacks(config):
    dataset_name = config.dataset_name
    data_path = config.data_path
    batch_size = config.batch_size
    image_nc = config.num_channels
    image_size = (config.img_height, config.img_width)
    target = config.target

    stage1_path = os.path.join(dirparent, "stage1/stage1_" + dataset_name + ".pt")
    if sys.argv[2] == "none":
        stage2_path = os.path.join(dirparent, "stage2/stage2_" + dataset_name + "_" + "abrupt" + ".pt")
    else:
        stage2_path = os.path.join(dirparent, "stage2/stage2_" + dataset_name + "_" + sys.argv[2] + ".pt")

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    target_models = []
    stage1_model = stage1().to(device)
    stage1_model.load_state_dict(torch.load(stage1_path))
    stage1_model.eval()
    stage2_model = stage2().to(device)
    stage2_model.load_state_dict(torch.load(stage2_path))
    stage2_model.eval()

    # root_dir = "../udacity-data"
    # attacks = ("FGSM", "Optimization", "Optimization Universal", "AdvGAN", "AdvGAN Universal")

    print("Loading training data...")
    X = np.load(data_path + "/X_train.npy")
    Y = pd.read_csv(data_path + "/Y_train_attack_" + sys.argv[2] + ".csv")
    # full_dataset = stage2_data(X, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=56)
    # train_dataset = stage2_data(X_train, Y_train)
    test_dataset = stage2_data(X_test, Y_test)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    num_sample = len(test_dataset)

    print("Attacking: RAIDS")
    # # fgsm attack
    # fgsm_result = []
    # fgsm_diff = []
    # fgsm_ast, diff, y_true, y_pred, y_pred_FGSM = fgsm_ex(
    #     test_data_loader,
    #     stage1_model,
    #     stage2_model,
    #     dataset_name,
    #     "FGSM",
    #     image_nc,
    #     target,
    #     device,
    #     num_sample,
    #     image_size,
    # )
    # result_print("FGSM", dataset_name, fgsm_ast, diff, fgsm_result, fgsm_diff, y_true, y_pred, y_pred_FGSM)

    # # optimization attack
    # opt_result = []
    # opt_diff = []
    # opt_ast, diff, y_true, y_pred, y_pred_opt = opt_ex(
    #     test_data_loader,
    #     stage1_model,
    #     stage2_model,
    #     dataset_name,
    #     "Opt",
    #     image_nc,
    #     target,
    #     device,
    #     num_sample,
    #     image_size,
    # )
    # result_print("Opt", dataset_name, opt_ast, diff, opt_result, opt_diff, y_true, y_pred, y_pred_opt)

    # # optimized-based universal attack
    # optu_result = []
    # optu_diff = []
    # optu_ast, diff, y_true, y_pred, y_pred_optU = opt_uni_ex(
    #     test_data_loader,
    #     stage1_model,
    #     stage2_model,
    #     dataset_name,
    #     "OptU",
    #     image_nc,
    #     target,
    #     device,
    #     num_sample,
    #     image_size,
    # )

    # result_print("OptU", dataset_name, optu_ast, diff, optu_result, optu_diff, y_true, y_pred, y_pred_optU)

    # advGAN attack
    advGAN_result = []
    advGAN_diff = []

    advGAN_ast, diff, y_true, y_pred, y_pred_advGAN = advGAN_ex(
        test_data_loader,
        stage1_model,
        stage2_model,
        dataset_name,
        "advGAN",
        image_nc,
        target,
        device,
        num_sample,
        image_size,
    )
    result_print("advGAN", dataset_name, advGAN_ast, diff, advGAN_result, advGAN_diff, y_true, y_pred, y_pred_advGAN)

    # advGAN_universal attack
    advGANU_result = []
    advGANU_diff = []

    advGANU_ast, diff, y_true, y_pred, y_pred_advGANU = advGANU_ex(
        test_data_loader,
        stage1_model,
        stage2_model,
        dataset_name,
        "advGANU",
        image_nc,
        target,
        device,
        num_sample,
        image_size,
    )
    result_print(
        "advGANU", dataset_name, advGANU_ast, diff, advGANU_result, advGANU_diff, y_true, y_pred, y_pred_advGANU
    )

    # save success rate
    success_rate = pd.DataFrame(columns=["FGSM", "Optimization", "OptimizationU", "AdvGAN", "AdvGANU"])
    success_rate.loc[0] = [fgsm_ast, opt_ast, optu_ast, advGAN_ast, advGANU_ast]
    success_rate.to_csv("./results/" + dataset_name + "_attack_success_rate_" + sys.argv[2] + ".csv")


def fgsm_ex(test_dataset, stage1, stage2, dataset_name, attack_name, image_nc, target, device, num_sample, image_size):
    print("testing", attack_name)
    fgsm_success = 0
    total_noise = 0
    diff_total = np.array([])
    # print(len(test_dataset))
    y_pred, y_true, y_pred_advGAN = [], [], []

    inter = 0.0
    for i, sample in enumerate(test_dataset):
        if i / len(test_dataset) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1
        if i % 64 == 0:
            plot_fig = True
        diff, pred_angle, pred_angle_attack, plt_, norm_noise, _ = fgsm_attack_test(
            stage1, sample[0], target, device, image_size, plot_fig
        )
        diff = np.squeeze(diff)
        diff_total = np.concatenate((diff_total, diff))
        if i % 64 == 0:
            Path("results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/").mkdir(
                parents=True, exist_ok=True
            )
            plt_.savefig(
                "results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/" + str(i) + ".jpg",
                bbox_inches="tight",
            )
        plt_.close()

        total_noise += norm_noise
        fgsm_success += len(diff[diff > abs(target)])

        # pass results to stage 2
        stage2_pred(device, stage2, pred_angle, pred_angle_attack, sample, y_true, y_pred, y_pred_advGAN)

    return (fgsm_success / num_sample), diff_total, y_true, y_pred, y_pred_advGAN


def opt_ex(test_dataset, stage1, stage2, dataset_name, attack_name, image_nc, target, device, num_sample, image_size):
    print("testing", attack_name)
    opt_success = 0
    diff_total = np.array([])
    y_pred, y_true, y_pred_advGAN = [], [], []

    inter = 0.0
    for i, sample in enumerate(test_dataset):
        if i / len(test_dataset) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1
        if i % 64 == 0:
            plot_fig = True
        diff, pred_angle, pred_angle_attack, plt_, _ = optimized_attack_test(
            stage1, sample[0], target, device, image_size, plot_fig
        )
        diff = np.squeeze(diff)
        diff_total = np.concatenate((diff_total, diff))
        if i % 64 == 0:
            Path("results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/").mkdir(
                parents=True, exist_ok=True
            )
            plt_.savefig(
                "results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/" + str(i) + ".jpg",
                bbox_inches="tight",
            )
        plt_.close()

        opt_success += len(diff[diff > abs(target)])
        # pass results to stage 2
        stage2_pred(device, stage2, pred_angle, pred_angle_attack, sample, y_true, y_pred, y_pred_advGAN)

    return (opt_success / num_sample), diff_total, y_true, y_pred, y_pred_advGAN


def opt_uni_ex(
    test_dataset, stage1, stage2, dataset_name, attack_name, image_nc, target, device, num_sample, image_size
):
    print("testing", attack_name)
    opt_uni_success = 0
    diff_total = np.array([])
    y_pred, y_true, y_pred_advGAN = [], [], []

    inter = 0.0
    for i, sample in enumerate(test_dataset):
        if i / len(test_dataset) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1

        batch_size = sample[0].size(0)
        noise = np.load("./models/" + dataset_name + "_" + attack_name + "_noise_seed.npy")
        noise = np.tile(noise, (batch_size, 1, 1, 1))
        noise = torch.from_numpy(noise).type(torch.FloatTensor).to(device)

        if i % 64 == 0:
            plot_fig = True
        diff, pred_angle, pred_angle_attack, plt_, _ = optimized_uni_test(
            stage1, sample[0], device, noise, image_size, plot_fig
        )
        diff = np.squeeze(diff)
        diff_total = np.concatenate((diff_total, diff))
        if i % 64 == 0:
            Path("results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/").mkdir(
                parents=True, exist_ok=True
            )
            plt_.savefig(
                "results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/" + str(i) + ".jpg",
                bbox_inches="tight",
            )
        plt_.close()

        opt_uni_success += len(diff[diff > abs(target)])
        # pass results to stage 2
        stage2_pred(device, stage2, pred_angle, pred_angle_attack, sample, y_true, y_pred, y_pred_advGAN)

    return (opt_uni_success / num_sample), diff_total, y_true, y_pred, y_pred_advGAN


def advGAN_ex(
    test_dataset, stage1, stage2, dataset_name, attack_name, image_nc, target, device, num_sample, image_size
):
    print("testing", attack_name)
    advGAN_success = 0
    diff_total = np.array([])
    advGAN_generator = Generator(image_nc, image_nc, attack_name).to(device)
    advGAN_generator.load_state_dict(torch.load("./models/" + dataset_name + "_" + attack_name + "_netG_epoch_60.pth"))
    advGAN_generator.eval()

    y_pred, y_true, y_pred_advGAN = [], [], []

    inter = 0.0
    for i, sample in enumerate(test_dataset):
        if i / len(test_dataset) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1

        if i % 64 == 0:
            plot_fig = True
        diff, pred_angle, pred_angle_advGAN, plt_, _ = advGAN_test(
            stage1, sample[0], advGAN_generator, device, image_size, plot_fig
        )
        diff = np.squeeze(diff)
        diff_total = np.concatenate((diff_total, diff))
        if i % 64 == 0:
            Path("results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/").mkdir(
                parents=True, exist_ok=True
            )
            plt_.savefig(
                "results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/" + str(i) + ".jpg",
                bbox_inches="tight",
            )
        plt_.close()
        advGAN_success += len(diff[diff >= abs(target)])

        # pass results to stage 2
        stage2_pred(device, stage2, pred_angle, pred_angle_advGAN, sample, y_true, y_pred, y_pred_advGAN)
    return (advGAN_success / num_sample), diff_total, y_true, y_pred, y_pred_advGAN


def advGANU_ex(
    test_dataset, stage1, stage2, dataset_name, attack_name, image_nc, target, device, num_sample, image_size
):
    print("testing", attack_name)
    advGAN_uni_success = 0
    diff_total = np.array([])
    advGAN_generator = Generator(image_nc, image_nc, attack_name).to(device)
    advGAN_generator.load_state_dict(torch.load("./models/" + dataset_name + "_" + attack_name + "_netG_epoch_60.pth"))
    advGAN_generator.eval()

    y_pred, y_true, y_pred_advGAN_uni = [], [], []

    inter = 0.0
    for i, sample in enumerate(test_dataset):
        if i / len(test_dataset) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1

        # creating the
        batch_size = sample[0].size(0)
        noise_seed = np.load("./models/" + dataset_name + "_" + attack_name + "_noise_seed.npy")
        noise_seed = np.tile(noise_seed, (batch_size, 1, 1, 1))
        noise = advGAN_generator(torch.from_numpy(noise_seed).type(torch.FloatTensor).to(device))

        if i % 64 == 0:
            plot_fig = True
        diff, pred_angle, pred_angle_advGAN_uni, plt_, _ = advGAN_uni_test(
            stage1, sample[0], device, noise, image_size, plot_fig
        )
        diff = np.squeeze(diff)
        diff_total = np.concatenate((diff_total, diff))
        if i % 64 == 0:
            Path("results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/").mkdir(
                parents=True, exist_ok=True
            )
            plt_.savefig(
                "results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/" + str(i) + ".jpg",
                bbox_inches="tight",
            )
        plt_.close()
        advGAN_uni_success += len(diff[diff >= abs(target)])

    # pass results to stage 2
    stage2_pred(device, stage2, pred_angle, pred_angle_advGAN_uni, sample, y_true, y_pred, y_pred_advGAN_uni)

    return (advGAN_uni_success / num_sample), diff_total, y_true, y_pred, y_pred_advGAN_uni


def stage2_pred(device, stage2, pred_angle, pred_angle_attack, sample, y_true, y_pred, y_pred_attack):
    # pass results to stage 2
    batch_x, angle, target = sample
    batch_x = batch_x.type(torch.FloatTensor)
    angle = angle.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)
    batch_x = batch_x.to(device)
    angle = angle.to(device)
    target = target.to(device)

    y_true.extend(target.data)

    output = stage2(torch.abs(torch.sub(angle.unsqueeze(-1), pred_angle)))
    pred = np.round(torch.sigmoid(output.detach()))
    y_pred.extend(pred.reshape(-1))

    output_advGAN = stage2(torch.abs(torch.sub(angle.unsqueeze(-1), pred_angle_attack)))
    pred = np.round(torch.sigmoid(output_advGAN.detach()))
    y_pred.extend(pred.reshape(-1))


def result(y_true, y_pred):
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred).flatten()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    # print("TN, FP, FN, TP :", cf_matrix)
    # print("accuracy :", accuracy)

    # Initialize a blank dataframe and keep adding
    df = pd.DataFrame(columns=["TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall"])
    df.loc[0] = cf_matrix.tolist() + [accuracy, precision, recall]
    df["Total_Actual_Neg"] = df["TN"] + df["FP"]
    df["Total_Actual_Pos"] = df["FN"] + df["TP"]
    df["Total_Pred_Neg"] = df["TN"] + df["FN"]
    df["Total_Pred_Pos"] = df["FP"] + df["TP"]
    df["TP_Rate"] = df["TP"] / df["Total_Actual_Pos"]  # Recall
    df["FP_Rate"] = df["FP"] / df["Total_Actual_Neg"]
    df["TN_Rate"] = df["TN"] / df["Total_Actual_Neg"]
    df["FN_Rate"] = df["FN"] / df["Total_Actual_Pos"]
    return df


def result_print(attack_name, dataset_name, ast, diff, attack_result, attack_diff, y_true, y_pred, y_pred_attack):
    print(f"Success Rate of {attack_name}:", ast)
    attack_result.append(ast)
    attack_diff.append(diff)

    result_org = result(y_true, y_pred)
    print("RAIDS without attack:")
    print(result_org)

    result_advGAN = result(y_true, y_pred_attack)
    print(f"RAIDS under attack by {attack_name}:")
    print(result_advGAN)

    df_all = pd.concat([result_org, result_advGAN])
    df_all.to_csv("./results/" + dataset_name + "_" + attack_name + "_" + sys.argv[2] + ".csv")


if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.attacksconfig()
    dirname = os.path.dirname(os.path.abspath(__file__))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    attacks(config)
