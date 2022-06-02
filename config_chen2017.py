# preprocess
class DataConfig(object):
    dataset_name = "chen2017"
    data_path = "/home/ubuntu/RAIDS2/data/chen2017/"
    data_name = "hsv_gray_diff_ch2"  # hsv_gray_diff_ch4
    img_height = 100
    img_width = 100
    num_channels = 2


# stage 1 training
class TrainConfig1(DataConfig):
    batch_size = 32
    num_epoch = 32


# stage 2 training
class TrainConfig2(DataConfig):
    batch_size = 32
    num_epoch = 32


# RAIDS both stage 1 and 2
class RAIDSconfig(DataConfig):
    batch_size = 32
    num_epoch = 50


# optiU training
class optiUConfig(DataConfig):
    batch_size = 32
    num_epoch = 60
    target = 0.3


# advGAN training
class advGANConfig(DataConfig):
    batch_size = 32
    num_epoch = 60
    target = 0.3


# attacks
class attacksconfig(DataConfig):
    batch_size = 32
    num_epoch = 60
    target = 0.3


# class TestConfig(TrainConfig2):
#     batch_size = 32
#     num_epoch = 15
#     model_path = "models/weights_hsv_gray_diff_ch4_comma_large_dropout-01-0.00540.hdf5"
#     angle_train_mean = -0.004179079
