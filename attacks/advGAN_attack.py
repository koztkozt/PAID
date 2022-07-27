from sklearn.model_selection import train_test_split
import os, sys, importlib
import numpy as np

np.random.seed(0)
import torch

torch.manual_seed(0)
import pandas as pd
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.advGAN.advGAN import AdvGAN_Attack
from attacks.advGAN.advGAN_Uni import AdvGAN_Uni_Attack

from stage1.data import stage1_data
from stage1.model import stage1


def advGAN_Attack(attack_name, target_model_path, target, train_dataset, config, universal=False):
    image_nc = config.num_channels
    epochs = config.num_epoch
    batch_size = config.batch_size
    dataset_name = config.dataset_name
    image_size = (config.img_height, config.img_width)

    BOX_MIN = 0
    BOX_MAX = 1
    # target = 0.2
    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    targeted_model = stage1().to(device)
    targeted_model.load_state_dict(torch.load(target_model_path))
    targeted_model.eval()

    if not universal:
        advGAN = AdvGAN_Attack(device, targeted_model, attack_name, target, image_nc, BOX_MIN, BOX_MAX, batch_size)
    else:
        advGAN = AdvGAN_Uni_Attack(
            device, targeted_model, attack_name, image_size, target, image_nc, BOX_MIN, BOX_MAX, batch_size
        )

    advGAN.train(train_dataset, epochs)
    return advGAN


if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.advGANConfig()
    dirname = os.path.dirname(os.path.abspath(__file__))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    attack_name = "advGAN"
    dataset_name = config.dataset_name
    data_path = config.data_path
    target = config.target
    target_model_path = os.path.join(dirparent, "stage1/stage1_" + sys.argv[3] + "_" + dataset_name + ".pt")

    print("Loading training data...")
    X_train = np.load(dirparent + "/" + data_path + "X_train.npy")
    Y_train = pd.read_csv(dirparent + "/" + data_path + "Y_train_attack_none.csv")
    train_dataset = stage1_data(X_train, Y_train)

    print("Start advGAN training")
    advGAN = advGAN_Attack(dataset_name, target_model_path, target, train_dataset, config)
    torch.save(advGAN.netG.state_dict(), "./models/" + dataset_name + "_" + attack_name + "_netG_epoch_60.pth")

    print("Start advGAN_uni training")
    advGAN_uni = advGAN_Attack(dataset_name, target_model_path, target, train_dataset, config, universal=True)
    advGAN_uni.save_noise_seed("./models/" + dataset_name + "_" + attack_name + "U_noise_seed.npy")
    torch.save(advGAN_uni.netG.state_dict(), "./models/" + dataset_name + "_" + attack_name + "U_netG_epoch_60.pth")

    print("Finish training")
