import os, sys, importlib
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        df = pd.read_csv("./results/score_" + dataset_name + "_" + sys.argv[2] + "_" + attack_names[i] + ".csv")
        df_detection = pd.DataFrame(columns=["orig_detected_bit", "adv_detected_bit", "orig_detected_blur", "adv_detected_blur","orig_detected_both", "adv_detected_both"])
        df_detection.index.rename('threshold', inplace=True)
        for threshold in np.arange(0.00,2.01,0.01):
            df_detection.loc[threshold, "orig_detected_bit"] = (df["score_bit"] > threshold).sum()
            df_detection.loc[threshold, "orig_detected_blur"] = (df["score_blur"] > threshold).sum()
            df_detection.loc[threshold, "orig_detected_both"] = (df["max_score"] > threshold).sum()

            df_detection.loc[threshold, "adv_detected_bit"] = (df["score_adv_bit"] > threshold).sum()
            df_detection.loc[threshold, "adv_detected_blur"] = (df["score_adv_blur"] > threshold).sum()
            df_detection.loc[threshold, "adv_detected_both"] = (df["max_score_adv"] > threshold).sum()
        
        num_sample = len(df)
        for detected in ["orig_detected_bit", "adv_detected_bit", "orig_detected_blur", "adv_detected_blur","orig_detected_both", "adv_detected_both"]:
             df_detection[detected+"_rate"] =  df_detection[detected] / num_sample

        print("Detection results")
        # print(df_detection)
        df_detection.to_csv("./results/" + dataset_name + "_" + sys.argv[2] + "_" + attack_names[i] + ".csv")
