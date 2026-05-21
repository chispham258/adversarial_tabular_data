import os
import pandas as pd

from tqdm import tqdm
from diagnostic_module.diagnosis_cls import Diagnosis
from tools.file import NeighborhoodParams

directory = "results_nn"
list_of_files = os.listdir(directory)

attacks = ["fgm", "pgd", "bim", "noise", "per", "zoo", "hsj", "lpf"]

for input_file in tqdm(list_of_files, desc="Dataset"):

    path_diag = os.path.join(directory, input_file, "diagnoses")
    path_quality = os.path.join(directory, input_file, "diag_quality")
    os.makedirs(path_diag, exist_ok=True)
    os.makedirs(path_quality, exist_ok=True)

    diagnosed_file = os.path.join(directory, input_file, "datasets", "org.csv")
    diagnosed_file_df = pd.read_csv(diagnosed_file)

    if isinstance(diagnosed_file_df, pd.DataFrame):

        try:
            diagnosis = Diagnosis(diagnosed_file_df)
            result_diagnose = diagnosis.diagnose(params=NeighborhoodParams(num_of_intervals=8, reducts=2500))

            result_diagnose.attributes_cont.to_csv(os.path.join(path_diag, "org.csv"), index=False)
            result_diagnose.quality_measures.to_csv(os.path.join(path_quality, "org.csv"), index=False)
        except:
            print(diagnosed_file)

        if isinstance(result_diagnose.attributes_cont, pd.DataFrame):

            for selected_attack in attacks:
                try:
                    monitor_file = os.path.join(directory, input_file, "datasets", selected_attack + ".csv")
                    result_monitor = diagnosis.monitor(pd.read_csv(monitor_file))

                    result_monitor.attributes_cont.to_csv(os.path.join(path_diag, selected_attack + ".csv"), index=False)
                except:
                   print(monitor_file)
