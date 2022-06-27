from sklearn.model_selection import train_test_split
import os, sys, importlib
import numpy as np
import pandas as pd

np.random.seed(0)

import torch

torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.optimization_attack import optimized_attack
from stage1.data import stage1_data
from stage1.model import stage1


def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    v_ = v.detach().cpu().numpy()
    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v_ = v_ * min(1, xi / np.linalg.norm(v_.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v_ = np.sign(v_) * np.minimum(abs(v_), xi)
    else:
        raise ValueError("Values of p different from 2 and Inf are currently not supported...")

    return torch.from_numpy(v_)


def universal_attack(dataset, model, device, target, delta=0.3, max_iters=np.inf, xi=10, p=np.inf, max_iter_lbfgs=30):
    v = 0
    fooling_rate = 0.0
    num_images = len(dataset)

    itr = 0
    while fooling_rate < 1 - delta and itr < max_iters:
        # np.random.shuffle(dataset)
        print("Starting pass number: ", itr)

        inter = 0.1
        for i, sample in enumerate(dataset):
            if i / len(dataset) > inter:
                inter += 0.1
                print(f"training completed: {(inter):.0%}")

            cur_img = sample[0]
            cur_img = cur_img.type(torch.FloatTensor)
            cur_img = cur_img.to(device)
            perturbed_image = cur_img + v
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            perturbed_image = perturbed_image.to(device)
            temp = is_adversary(model, cur_img, perturbed_image, target)
            if not temp[0]:
                # print('>> k = ', k, ', pass #', itr)
                _, d_noise, _, _ = optimized_attack(model, temp[1], perturbed_image, device)
                v = v + d_noise
                v = proj_lp(v, xi, p)
                v = v.to(device)
        itr += 1

        count = 0
        for i, sample in enumerate(dataset):
            cur_img = sample[0]
            cur_img = cur_img.type(torch.FloatTensor)
            cur_img = cur_img.to(device)
            perturbed_image = cur_img + v
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

            perturbed_image = perturbed_image.to(device)
            if is_adversary(model, cur_img, perturbed_image, target)[0]:
                count += 1

        fooling_rate = count / num_images

        print("Fooling rate: ", fooling_rate)
    # demension of v : (1, 3, image_size)
    return v


def is_adversary(model, x, x_adv, target):
    # print(target)
    y_pred = model(x).item()
    y_adv = model(x_adv).item()
    # print(y_pred, y_adv)
    if abs(y_adv - y_pred) >= abs(target):
        return [True]
    else:
        return [False, target - abs(y_adv - y_pred)]


def generate_noise(dataset, model, device, target):
    perturbation = universal_attack(dataset, model, device, target)
    perturbation = perturbation.detach().cpu().numpy()
    return perturbation


if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.optiUConfig()
    dirname = os.path.dirname(os.path.abspath(__file__))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    attack_name = "OptU"
    dataset_name = config.dataset_name
    data_path = config.data_path
    target = config.target
    batch_size = config.batch_size
    target_model_path = os.path.join(dirparent, "stage1/stage1_" + dataset_name + ".pt")

    print("Loading training data...")
    X = np.load(dirparent + "/" + data_path + "X_train.npy")
    Y = pd.read_csv(dirparent + "/" + data_path + "Y_train.csv")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=56)
    train_dataset = stage1_data(X_train, Y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    print("Length of dataloader :", len(train_dataloader))

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    targeted_model = stage1().to(device)
    targeted_model.load_state_dict(torch.load(target_model_path))
    targeted_model.eval()

    print("Start optiU training")
    perturbation = generate_noise(train_dataloader, targeted_model, device, target)
    np.save("./models/" + dataset_name + "_" + attack_name + "_noise_seed.npy", perturbation)
    print("Finish optiU training.")
