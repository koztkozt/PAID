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

    stage1_path = os.path.join(dirparent, "stage1/stage1_" + sys.argv[3] + "_" + dataset_name + ".pt")
    if sys.argv[2] == "none":
        stage2_path = os.path.join(dirparent, "stage2/stage2_" + sys.argv[3] + "_" + dataset_name + "_" + "abrupt" + ".pt")
    else:
        stage2_path = os.path.join(dirparent, "stage2/stage2_" + sys.argv[3] + "_" + dataset_name + "_" + sys.argv[2] + ".pt")

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    if torch.cuda.is_available():
        target_models = []
        stage1_model = stage1().to(device)
        stage1_model.load_state_dict(torch.load(stage1_path))
        stage1_model.eval()
        stage2_model = stage2().to(device)
        stage2_model.load_state_dict(torch.load(stage2_path))
        stage2_model.eval()
    else:
        target_models = []
        stage1_model = stage1().to(device)
        stage1_model.load_state_dict(torch.load(stage1_path,map_location=torch.device('cpu')))
        stage1_model.eval()
        stage2_model = stage2().to(device)
        stage2_model.load_state_dict(torch.load(stage2_path,map_location=torch.device('cpu')))
        stage2_model.eval()

    # root_dir = "../udacity-data"
    # attacks = ("FGSM", "Optimization", "Optimization Universal", "AdvGAN", "AdvGAN Universal")

    print("Loading testing data...")
    X_test = np.load(dirparent + "/" + data_path +"X_test.npy")
    Y_test = pd.read_csv(dirparent + "/" + data_path +"Y_test_attack_" + sys.argv[2] + ".csv")
    test_dataset = stage2_data(X_test, Y_test)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    num_sample = len(test_dataset)
    print("Length of dataloader :", len(test_data_loader))
    
    image_number = 2112
    print("Attacking: RAIDS")
    # # fgsm attack
    # fgsm_result = []
    # fgsm_diff = []
    # fgsm_ast, diff, y_true, y_pred, y_pred_FGSM, fgsm_robustness = fgsm_ex(
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
    # print('robustness', fgsm_robustness)
    # save_attack_success(dataset_name,"FGSM",fgsm_ast)
    # result_print("FGSM", dataset_name, fgsm_ast, diff, fgsm_result, fgsm_diff, y_true, y_pred, y_pred_FGSM)

    # # optimization attack
    # opt_result = []
    # opt_diff = []
    # opt_ast, diff, y_true, y_pred, y_pred_opt, opt_robustness = opt_ex(
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
    # print('robustness', opt_robustness)
    # save_attack_success(dataset_name,"Opt",opt_ast)
    # result_print("Opt", dataset_name, opt_ast, diff, opt_result, opt_diff, y_true, y_pred, y_pred_opt)

    # # optimized-based universal attack
    # optu_result = []
    # optu_diff = []
    # optu_ast, diff, y_true, y_pred, y_pred_optU, optu_robustness = opt_uni_ex(
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
    # print('robustness', optu_robustness)
    # save_attack_success(dataset_name,"OptU",optu_ast)
    # result_print("OptU", dataset_name, optu_ast, diff, optu_result, optu_diff, y_true, y_pred, y_pred_optU)

    # advGAN attack
    advGAN_result = []
    advGAN_diff = []

    advGAN_ast, diff, y_true, y_pred, y_pred_advGAN, advGAN_robustness = advGAN_ex(
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
    print('robustness', advGAN_robustness)
    save_attack_success(dataset_name,"advGAN",advGAN_ast)
    result_print("advGAN", dataset_name, advGAN_ast, diff, advGAN_result, advGAN_diff, y_true, y_pred, y_pred_advGAN)

#     # advGAN_universal attack
#     advGANU_result = []
#     advGANU_diff = []

#     advGANU_ast, diff, y_true, y_pred, y_pred_advGANU, advGANU_robustness = advGANU_ex(
#         test_data_loader,
#         stage1_model,
#         stage2_model,
#         dataset_name,
#         "advGANU",
#         image_nc,
#         target,
#         device,
#         num_sample,
#         image_size,
#     )
#     print('robustness', advGANU_robustness)
#     save_attack_success(dataset_name,"advGANU",advGANU_ast)
#     result_print(
#         "advGANU", dataset_name, advGANU_ast, diff, advGANU_result, advGANU_diff, y_true, y_pred, y_pred_advGANU
#     )

    # save success rate
    success_rate = pd.DataFrame(columns=["FGSM", "Optimization", "OptimizationU", "AdvGAN", "AdvGANU"])
    success_rate.loc[0] = [fgsm_ast, opt_ast, optu_ast, advGAN_ast, advGANU_ast]
    success_rate.to_csv("./results/" + dataset_name + "_attack_success_rate_" + sys.argv[2] + ".csv")

    # save robustness
    robustness = pd.DataFrame(columns=["FGSM", "Optimization", "OptimizationU", "AdvGAN", "AdvGANU"])
    robustness.loc[0] = [fgsm_robustness, opt_robustness, optu_robustness, advGAN_robustness, advGANU_robustness]
    robustness.to_csv("./results/" + dataset_name + "_robustness_" + sys.argv[2] + ".csv")
    
def fgsm_ex(test_dataset, stage1, stage2, dataset_name, attack_name, image_nc, target, device, num_sample, image_size):
    print("testing", attack_name)
    fgsm_success = 0
    diff_total = []
    total_noise = []
    x = []
    # print(len(test_dataset))
    y_pred, y_true, y_pred_attack = [], [], []

    inter = 0.0
    for i, sample in enumerate(test_dataset):
        if i != image_number:
            continue
        if i / len(test_dataset) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1
        if i % 64 == 0:
            plot_fig = True

        diff, pred_angle, pred_angle_attack, plt_, noise,perturbed_image  = fgsm_attack_test(
            stage1, sample[0], target, device, image_size, plot_fig
        )
        diff_total.extend(diff)
        np.save('report_image/perturbed_image_' + attack_name, perturbed_image)
        
        if i % 64 == 0:
            Path("i" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/").mkdir(
                parents=True, exist_ok=True
            )
            plt_.savefig(
                "results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/" + str(i) + ".jpg",
                bbox_inches="tight",
            )
        plt_.close()
        
        if abs(diff) > abs(target):
            fgsm_success += 1
            total_noise.append(noise)
            x.append(sample[0].detach().cpu().numpy())
                 
        # pass results to stage 2
        stage2_pred(device, stage2, pred_angle, pred_angle_attack, sample, y_true, y_pred, y_pred_attack)

    robustness = empirical_robustness(np.array(x),np.array(total_noise))
    
    return (fgsm_success / num_sample), diff_total, y_true, y_pred, y_pred_attack, robustness


def opt_ex(test_dataset, stage1, stage2, dataset_name, attack_name, image_nc, target, device, num_sample, image_size):
    print("testing", attack_name)
    opt_success = 0
    diff_total = []
    total_noise = []
    x = []    
    y_pred, y_true, y_pred_attack = [], [], []

    inter = 0.0
    for i, sample in enumerate(test_dataset):
        if i != image_number:
            continue
        if i / len(test_dataset) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1
        if i % 64 == 0:
            plot_fig = True
        diff, pred_angle, pred_angle_attack, plt_, noise,perturbed_image  = optimized_attack_test(
            stage1, sample[0], target, device, image_size, plot_fig
        )
        diff_total.extend(diff)
        np.save('report_image/perturbed_image_' + attack_name, perturbed_image)
        if i % 64 == 0:
            Path("results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/").mkdir(
                parents=True, exist_ok=True
            )
            plt_.savefig(
                "results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/" + str(i) + ".jpg",
                bbox_inches="tight",
            )
        plt_.close()

        if abs(diff) > abs(target):
            opt_success += 1
            total_noise.append(noise)
            x.append(sample[0].detach().cpu().numpy())
            
        # pass results to stage 2
        stage2_pred(device, stage2, pred_angle, pred_angle_attack, sample, y_true, y_pred, y_pred_attack)
    
    robustness = empirical_robustness(np.array(x),np.array(total_noise))
    
    return (opt_success / num_sample), diff_total, y_true, y_pred, y_pred_attack, robustness


def opt_uni_ex(
    test_dataset, stage1, stage2, dataset_name, attack_name, image_nc, target, device, num_sample, image_size
):
    print("testing", attack_name)
    opt_uni_success = 0
    diff_total = []
    total_noise = []
    x = []
    y_pred, y_true, y_pred_attack = [], [], []

    inter = 0.0
    for i, sample in enumerate(test_dataset):
        if i != image_number:
            continue
        if i / len(test_dataset) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1

        batch_size = sample[0].size(0)
        noise = np.load("./models/" + dataset_name + "_" + attack_name + "_noise_seed.npy")
        noise = np.tile(noise, (batch_size, 1, 1, 1))
        noise = torch.from_numpy(noise).type(torch.FloatTensor).to(device)

        if i % 64 == 0:
            plot_fig = True
        diff, pred_angle, pred_angle_attack, plt_,perturbed_image  = optimized_uni_test(
            stage1, sample[0], device, noise, image_size, plot_fig
        )
        diff_total.extend(diff)
        np.save('report_image/perturbed_image_' + attack_name, perturbed_image)
        if i % 64 == 0:
            Path("results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/").mkdir(
                parents=True, exist_ok=True
            )
            plt_.savefig(
                "results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/" + str(i) + ".jpg",
                bbox_inches="tight",
            )
        plt_.close()

        if abs(diff) > abs(target):
            opt_uni_success += 1
            total_noise.append(noise.detach().cpu().numpy())
            x.append(sample[0].detach().cpu().numpy())
            
        # pass results to stage 2
        stage2_pred(device, stage2, pred_angle, pred_angle_attack, sample, y_true, y_pred, y_pred_attack)
    
    robustness = empirical_robustness(np.array(x),np.array(total_noise))
    
    return (opt_uni_success / num_sample), diff_total, y_true, y_pred, y_pred_attack, robustness


def advGAN_ex(
    test_dataset, stage1, stage2, dataset_name, attack_name, image_nc, target, device, num_sample, image_size
):
    print("testing", attack_name)
    advGAN_success = 0
    total_noise = []
    x = []
    diff_total = []
    advGAN_generator = Generator(image_nc, image_nc, attack_name).to(device)
    if torch.cuda.is_available():
        advGAN_generator.load_state_dict(torch.load("./models/" + dataset_name + "_" + attack_name + "_netG_epoch_60.pth"))
    else:
        advGAN_generator.load_state_dict(torch.load("./models/" + dataset_name + "_" + attack_name + "_netG_epoch_60.pth",map_location=torch.device('cpu')))
    advGAN_generator.eval()

    y_pred, y_true, y_pred_attack = [], [], []

    inter = 0.0
    for i, sample in enumerate(test_dataset):
        if i != image_number:
            continue
        if i / len(test_dataset) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1

        if i % 64 == 0:
            plot_fig = True
        diff, pred_angle, pred_angle_advGAN, plt_, noise,perturbed_image  = advGAN_test(
            stage1, sample[0], advGAN_generator, device, image_size, plot_fig
        )
        diff_total.extend(diff)
        np.save('report_image/perturbed_image_none', sample[0])
        np.save('report_image/perturbed_image_' + attack_name, perturbed_image)
        if i % 64 == 0:
            Path("results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/").mkdir(
                parents=True, exist_ok=True
            )
            plt_.savefig(
                "results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/" + str(i) + ".jpg",
                bbox_inches="tight",
            )
        plt_.close()
        if abs(diff) > abs(target):
            advGAN_success += 1
            total_noise.append(noise)
            x.append(sample[0].detach().cpu().numpy())
            
        # pass results to stage 2
        stage2_pred(device, stage2, pred_angle, pred_angle_advGAN, sample, y_true, y_pred, y_pred_attack)
    
    robustness = empirical_robustness(np.array(x),np.array(total_noise))
    
    return (advGAN_success / num_sample), diff_total, y_true, y_pred, y_pred_attack, robustness


def advGANU_ex(
    test_dataset, stage1, stage2, dataset_name, attack_name, image_nc, target, device, num_sample, image_size
):
    print("testing", attack_name)
    advGAN_uni_success = 0
    total_noise = []
    x = []
    diff_total = []
    advGAN_generator = Generator(image_nc, image_nc, attack_name).to(device)
    if torch.cuda.is_available():
        advGAN_generator.load_state_dict(torch.load("./models/" + dataset_name + "_" + attack_name + "_netG_epoch_60.pth"))
    else:
        advGAN_generator.load_state_dict(torch.load("./models/" + dataset_name + "_" + attack_name + "_netG_epoch_60.pth",map_location=torch.device('cpu')))
    advGAN_generator.eval()

    y_pred, y_true, y_pred_attack_uni = [], [], []

    inter = 0.0
    for i, sample in enumerate(test_dataset):
        if i != image_number:
            continue
        
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
        diff, pred_angle, pred_angle_advGAN_uni, plt_,perturbed_image  = advGAN_uni_test(
            stage1, sample[0], device, noise, image_size, plot_fig
        )
        diff_total.extend(diff)
        np.save('report_image/perturbed_image_' + attack_name, perturbed_image)
        if i % 64 == 0:
            Path("results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/").mkdir(
                parents=True, exist_ok=True
            )
            plt_.savefig(
                "results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/" + str(i) + ".jpg",
                bbox_inches="tight",
            )
        plt_.close()
        if abs(diff) > abs(target):
            advGAN_uni_success += 1
            total_noise.append(noise.detach().cpu().numpy())
            x.append(sample[0].detach().cpu().numpy())

        # pass results to stage 2
        stage2_pred(device, stage2, pred_angle, pred_angle_advGAN_uni, sample, y_true, y_pred, y_pred_attack_uni)
    
    robustness = empirical_robustness(np.array(x),np.array(total_noise))
    
    return (advGAN_uni_success / num_sample), diff_total, y_true, y_pred, y_pred_attack_uni, robustness


def stage2_pred(device, stage2, pred_angle, pred_angle_attack, sample, y_true, y_pred, y_pred_attack):
    # pass results to stage 2
    batch_x, angle, target = sample
    batch_x = batch_x.type(torch.FloatTensor)
    angle = angle.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)
    batch_x = batch_x.to(device)
    angle = angle.to(device)
    target = target.to(device)

    y_true.extend(target.detach().cpu().data)

    output = stage2(torch.abs(torch.sub(angle.unsqueeze(-1), pred_angle)))
    pred = np.round(torch.sigmoid(output.detach().cpu()))
    y_pred.extend(pred.reshape(-1))

    output_attack = stage2(torch.abs(torch.sub(angle.unsqueeze(-1), pred_angle_attack)))
    pred = np.round(torch.sigmoid(output_attack.detach().cpu()))
    y_pred_attack.extend(pred.reshape(-1))


def result(y_true, y_pred):
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred).flatten()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print("TN, FP, FN, TP :", cf_matrix)
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

def save_attack_success(dataset_name,attack_name, ast):
    # print("attack_success_rate", ast)
    with open("results/"+dataset_name+"_"+attack_name + "_attack_success.txt", "w") as f:
        f.write(str(ast))

def empirical_robustness(x, noise):
    # https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/e2a7b4ae5a4cefc1b17c3c2b82cdca455736d45f/art/metrics/metrics.py#L82
    norm_type = 2
    perts_norm = np.linalg.norm(noise.reshape(x.shape[0], -1), ord=norm_type, axis=1)
    robustness = np.mean(perts_norm / np.linalg.norm(x.reshape(x.shape[0], -1), ord=norm_type, axis=1))

    return robustness

    
if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.attacksconfig()
    dirname = os.path.dirname(os.path.abspath(__file__))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    attacks(config)
