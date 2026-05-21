import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

diag_final = pd.DataFrame()

directory = "results"
list_of_files = os.listdir(directory)

for input_file in tqdm(list_of_files, desc="Dataset"):

    path_diag = os.path.join(directory, input_file, "diagnoses")
    path_src = os.path.join(directory, input_file, "attacks")
    list_of_diagnoses = os.listdir(path_diag)

    for file in list_of_diagnoses:

        file_split = file.replace('.csv', '').split("_")

        df_diag = pd.read_csv(os.path.join(path_diag, file))
        df_source = pd.read_csv(os.path.join(path_src, file))

        bacc_test = balanced_accuracy_score(df_source[df_source["is_train"] == 0]["target"],
                                            df_source[df_source["is_train"] == 0]["prediction"])

        n_test = df_source[df_source["is_train"] == 0].shape[0]
        n_classes = len(np.unique(df_source["target"]))

        df_diag["dataset"] = file_split[0]
        df_diag["model"] = file_split[1]
        df_diag["attack"] = file_split[2]
        df_diag["n_test"] = n_test
        df_diag["n_classes"] = n_classes
        df_diag["bacc_test"] = bacc_test

        diag_final = pd.concat([diag_final, df_diag])

diag_final.to_csv("results/attacks_diagnoses.csv", index=False)
