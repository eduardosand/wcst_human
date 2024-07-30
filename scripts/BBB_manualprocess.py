# This script exists to load in processed data and comb through it to find reference wire
# for later processing
import numpy as np
import os
from pathlib import Path
from intracranial_ephys_utils.load_data import read_file, get_file_info
from intracranial_ephys_utils.manual_process import data_clean_viewer
from intracranial_ephys_utils.preprocess import make_trialwise_data, smooth_data
from behavior_analysis import process_wcst_behavior
import re
import pandas as pd

subject = 'IR95'
session = 'sess-3'
task = 'wcst'


timestamps_file = f"sub-{subject}-{session}-ph_timestamps.csv"

data_directory = Path(f'{os.pardir}/data/{subject}/{session}')
ph_file_path = get_file_info(data_directory / Path('raw'), 'photo1', '.ncs')

running_avg = 5
bhv_directory = data_directory / Path("behavior")
bhv_file_path = get_file_info(bhv_directory, f'{subject}', '.csv')

beh_data, rule_shifts_ind, _ = process_wcst_behavior(bhv_file_path,
                                                         running_avg=running_avg)

beh_data.set_index(['trial'], inplace=True)
beh_timestamps = pd.read_csv(bhv_directory / timestamps_file)

# global t-start
reader = read_file(ph_file_path)
reader.parse_header()
start_record = reader.global_t_start

bp = 1000
processed_data_directory = data_directory / 'preprocessed'
dataset_path = processed_data_directory / f'{subject}_{session}_{task}_lowpass_{bp}.npz'
full_dataset = np.load(dataset_path)
print('dataset loaded')
dataset = full_dataset['dataset'][:-1, :]
timestamps = full_dataset['dataset'][-1, :] # this should be in seconds from start of recording(start of file)
fs = full_dataset['eff_fs'][:-1]
electrode_names = full_dataset['electrode_names'][:-1]
# issue with electrode_names is sometimes there's an extra ending to denote the recording number, it should be
# present on all channels so we get rid of it by using the first name
stems = electrode_names[0].split("_")

# Now we have the dataset, next step is to rereference
electrode_names_str = " ".join(electrode_names)
electrode_names_str = re.sub(f"_{stems[-1]}", "", electrode_names_str)
electrode_names = np.array(electrode_names_str.split(" "))
# electrode_names_str = re.sub("m.... ", "", electrode_names_str)
electrode_names_fixed = re.sub("\d+", "", electrode_names_str)
skippables = ['spk', 'mic', 'eye']

data_clean_viewer(subject, session, task, data_directory, electrode_names, dataset, int(fs[0]))
