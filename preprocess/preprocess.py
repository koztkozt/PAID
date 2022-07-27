from skimage.exposure import rescale_intensity
from tensorflow.keras.utils import load_img, img_to_array
import pandas as pd
import numpy as np
from matplotlib.colors import rgb_to_hsv
import os, sys, importlib


# https://github.com/udacity/self-driving-car/blob/master/steering-models/community-models/rambo/preprocess_train_data.py
# The raw images are of size 640 x 480. In our final model, we resized the images to 256 x 192, converted from RGB color format to grayscale, computed lag 1 differences between frames and used 2 consecutive differenced images. For example, at time t we used [x_{t} - x_{t-1}, x_{t-1} - x_{t-2}] as input where x corresponds to the grayscale image. No future frames were used to predict the current steering angle.

def make_hsv_grayscale_diff_data(path, num_channels=2):
    df = pd.read_csv(path)
    # print("df: ", df)
    num_rows = df.shape[0]
    # print("num_rows: ", num_rows)
    # Creating the X array filled with zeros to store results later
    X = np.zeros((num_rows - num_channels, num_channels, row, col), dtype=np.float32)
    # X = np.zeros((num_rows - 21, num_channels, row, col), dtype=np.float32)

    for i in range(num_channels, num_rows):
    # for i in range(21, num_rows):
        if i % 1000 == 0:
            print("Processed " + str(i) + " images...")
        for j in range(num_channels):
            # print("i: ", i)
            # print("j: ", j)
            path0 = df["filename"].iloc[i - j - 1]
            path1 = df["filename"].iloc[i - j]
            # print("path0: ", df['fullpath'])

            # Loads an image into PIL format.
            img0 = load_img(data_path + "image/" + path0, target_size=(row, col))
            img1 = load_img(data_path + "image/" + path1, target_size=(row, col))
            # Converts a PIL Image instance to a Numpy array.
            img0 = img_to_array(img0)
            img1 = img_to_array(img1)
            # Convert float rgb values (in the range [0, 1]), in a numpy array to hsv values.
            img0 = rgb_to_hsv(img0)
            img1 = rgb_to_hsv(img1)

            # Subtracting an image from another image results in an image with the differences between the two.
            # You can just select the V channel as your grayscale image by splitting the HSV image in 3 and taking the 3rd channel
            img = img1[:, :, 2] - img0[:, :, 2]

            # Return image after stretching or shrinking its intensity levels.
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 1))

            # save all images into numpy
            X[i - num_channels, j, :, :] = img
            # X[i - 21, j, :, :] = img
            # print("num_channels: ", num_channels)
            # print("np.array(df[""].iloc[num_channels:]): ", df["angle"])

    # save all angles in radian as Y
    # Y = np.array(df["angle_convert"].iloc[num_channels:])
    # Y = df.iloc[num_channels:]
    # Y = df.iloc[21:]
    return X


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.DataConfig()
    data_path = config.data_path
    row, col = config.img_height, config.img_width
    dataset_name = config.dataset_name
    num_channels= config.num_channels
    
    if sys.argv[2] == "images": 
        print("Pre-processing images...")
        # 1. need to prepare the csv file first from original dataset
        # 2. need to put the picture and CSV files in the same folder
        # see https://github.com/cd-wang/RAIDS/issues/2
        X_all = make_hsv_grayscale_diff_data("{}data.csv".format(data_path), num_channels)
        np.save("{}X_all".format(data_path), X_all)
        np.save("{}X_all_mean".format(data_path), X_all.mean())
        print("Pre-processing data done")
        
    print("Pre-processing data...")   
    df = pd.read_csv("{}data.csv".format(data_path))
    Y_all = df.iloc[num_channels:]
    Y_all.to_csv(data_path + "Y_all.csv", index=False)
    print("Pre-processing data done")