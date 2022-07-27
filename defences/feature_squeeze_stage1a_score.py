import os, sys, importlib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,f1_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def results(y_true, y_pred):
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred).flatten()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # print("TN, FP, FN, TP :", cf_matrix)
    # print("accuracy :", accuracy)

    # Initialize a blank dataframe and keep adding
    df = pd.DataFrame(columns=["TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "F1"])
    df.loc[1] = cf_matrix.tolist() + [accuracy, precision, recall, f1]
    df["Total_Actual_Neg"] = df["TN"] + df["FP"]
    df["Total_Actual_Pos"] = df["FN"] + df["TP"]
    df["Total_Pred_Neg"] = df["TN"] + df["FN"]
    df["Total_Pred_Pos"] = df["FP"] + df["TP"]
    df["TP_Rate"] = df["TP"] / df["Total_Actual_Pos"]  # Recall
    df["FP_Rate"] = df["FP"] / df["Total_Actual_Neg"]
    df["TN_Rate"] = df["TN"] / df["Total_Actual_Neg"]
    df["FN_Rate"] = df["FN"] / df["Total_Actual_Pos"]        
    # print(df.tail())
    # df.to_csv("./results_fs_1a/accurracy_" +  sys.argv[3] + "_" + "udacity" + "_" + attack_name + ".csv")
    # print(df.at[1,'Accuracy'])
    return df.at[1,'Accuracy'], df.at[1,'F1']
    
if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.defencesconfig()
    dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    batch_size = config.batch_size
    data_path = config.data_path
    dataset_name = config.dataset_name

    attack_names = ["fgsm", "opt", "OptU", "advGAN", "advGANU"]
    # attack_names = ["advGANU"]
    for i in range(len(attack_names)):
        df = pd.read_csv("./results_fs_1a/score_" + dataset_name + "_" + sys.argv[2] + "_" + attack_names[i] + ".csv")
        df['score_both'] = df[["score_bit", "score_blur"]].max(axis=1)
        df_detection = pd.DataFrame(columns=["accuracy_bit", "accuracy_blur", "accuracy_both"])
        df_detection.index.rename('threshold', inplace=True)
        for threshold in np.arange(0.00,2.01,0.01):
            df['y_pred_bit'] = df['score_bit'] > threshold
            df['y_pred_blur'] = df['score_blur'] > threshold
            df['y_pred_both'] = df['score_both'] > threshold
            
            df_detection.loc[threshold, "accuracy_bit"], df_detection.loc[threshold, "F1_bit"] =results(df['anomaly'], df['y_pred_bit'])
            df_detection.loc[threshold, "accuracy_blur"], df_detection.loc[threshold, "F1_blur"] =results(df['anomaly'], df['y_pred_blur'])
            df_detection.loc[threshold, "accuracy_both"], df_detection.loc[threshold, "F1_both"] =results(df['anomaly'], df['y_pred_both'])

        print("Detection results")
        # print(df_detection)
        df_detection.to_csv("./results_fs_1a/accuracy_" + dataset_name + "_" + sys.argv[2] + "_" + attack_names[i] + ".csv")
