import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from behavior_analysis import process_wcst_behavior, plot_subject_performance
from pathlib import Path
import json

# summary_directory = "/home/eduardo/WCST_Human/Summary/U19_WCST_Progress.csv"
results_directory = Path(f"{os.pardir}/results")
subject = 'BERK01'
session = 'sess-1'
file_name = "BERK01_Wisconsin Card Sorting Eduardo_2024_Apr_05_0019.csv"
# file_name = "IR95_wcst6_2020_Aug_20_1802.csv"
# file_name = f'sub-{subject}-sess-{session}-beh.csv'
running_avg = 4
# data_directory = f"/home/eduardo/WCST_Human/{subject}/sess-{session}/behavior"
# file_path = os.path.join(data_directory, file_name)
# print(data_directory)
file_path = os.path.join(Path(f"{os.pardir}/data/{subject}/{session}/behavior/{file_name}"))
beh_data, rule_shifts_ind, _ = process_wcst_behavior(file_path, running_avg)
sbj_deets_path = Path(f'subject_deets.json')
sort_directory = Path(f"{os.pardir}/data/{subject}/{session}/spike_sorting")
# Open our file
with open(sbj_deets_path, 'r') as jsonfile:
    sbj_deets_data = json.load(jsonfile)


plot_subject_performance(beh_data['trial'],
                         beh_data[f'running_avg_{running_avg}'],
                         rule_shifts_ind, subject, session,
                         output_folder=results_directory)
plot_subject_performance(beh_data['trial'],
                         beh_data['response_time'],
                         rule_shifts_ind, subject, session,
                         output_folder=results_directory, save='rt')
