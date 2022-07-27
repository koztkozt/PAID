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
    X = np.load(data_path + "X_all.npy")
    Y_none = pd.read_csv(data_path + "Y_all_attack_none.csv")
    Y_abrupt = pd.read_csv(data_path + "Y_all_attack_abrupt.csv")
    Y_directed = pd.read_csv(data_path + "Y_all_attack_directed.csv")

    # In the first step we will split the data in training and remaining dataset
    X_train, X_rem, Y_none_train, Y_none_rem, Y_abrupt_train, Y_abrupt_rem, Y_directed_train, Y_directed_rem = train_test_split(X, Y_none, Y_abrupt, Y_directed, test_size=0.3, random_state=6)
    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    X_valid, X_test, Y_none_valid, Y_none_test, Y_abrupt_valid, Y_abrupt_test, Y_directed_valid, Y_directed_test = train_test_split(X_rem,Y_none_rem,Y_abrupt_rem,Y_directed_rem, test_size=0.5, random_state=6)

    np.save(data_path + "X_train",X_train)
    Y_none_train.to_csv(data_path + "Y_train_attack_none.csv", index=False)
    Y_abrupt_train.to_csv(data_path + "Y_train_attack_abrupt.csv", index=False)
    Y_directed_train.to_csv(data_path + "Y_train_attack_directed.csv", index=False)

    np.save(data_path + "X_valid",X_valid)
    Y_none_valid.to_csv(data_path + "Y_valid_attack_none.csv", index=False)
    Y_abrupt_valid.to_csv(data_path + "Y_valid_attack_abrupt.csv", index=False)
    Y_directed_valid.to_csv(data_path + "Y_valid_attack_directed.csv", index=False)

    np.save(data_path + "X_test",X_test)
    Y_none_test.to_csv(data_path + "Y_test_attack_none.csv", index=False)
    Y_abrupt_test.to_csv(data_path + "Y_test_attack_abrupt.csv", index=False)
    Y_directed_test.to_csv(data_path + "Y_test_attack_directed.csv", index=False)    
    print("Splittintg done")  