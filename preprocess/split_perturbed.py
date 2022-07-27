import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os, sys, importlib



if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.DataConfig()
    data_path = config.data_path
    row, col = config.img_height, config.img_width
    dataset_name = config.dataset_name

    print("Splittintg data...")    
    X = np.load(data_path + "X_all_perturbed.npy")
    Y = pd.read_csv(data_path + "Y_all_perturbed.csv")
    # In the first step we will split the data in training and remaining dataset
    X_train, X_rem, Y_train, Y_rem = train_test_split(X, Y, test_size=0.3, random_state=56)
    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    X_valid, X_test, Y_valid, Y_test  = train_test_split(X_rem,Y_rem, test_size=0.5, random_state=56)

    np.save(data_path + "X_train_perturbed",X_train)
    Y_train.to_csv(data_path + "Y_train_perturbed.csv", index=False)

    np.save(data_path + "X_valid_perturbed",X_valid)
    Y_valid.to_csv(data_path + "Y_valid_perturbed.csv", index=False)

    np.save(data_path + "X_test_perturbed",X_test)
    Y_test.to_csv(data_path + "Y_test_perturbed.csv", index=False)
  