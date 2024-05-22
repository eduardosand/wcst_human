# This function serves to grab the log file,
from pathlib import Path
import os
subject = "IR95"
session = "sess-3"

bhv_directory = Path(f"{os.pardir}/data/{subject}/{session}/behavior")
# print(os.listdir(bhv_directory))
files = os.listdir(bhv_directory)
bhv_files = [file for file in files if file.endswith(".log")]
# print(bhv_files)


# read in log file then split it up line by line, then throw out anything that wasn't a keypress
with open(bhv_directory / bhv_files[0]) as my_file:
    file_txt = my_file.read()
    all_lines = file_txt.split("\n")
    key_press_lines = [line for line in all_lines if "DATA" in line]

    # throw away the first key press (space bar to start the task) and the last (escape key to leave task)
    key_press_lines = key_press_lines[1:-1]
    response_times = [float(line.split(" ")[0]) for line in key_press_lines]
    new_trial_lines = [line for line in all_lines if "New trial " in line]
    print(new_trial_lines)
    trial_onset_times = [float(line.split(" ")[0]) for line in new_trial_lines]
    print(response_times)

# Next we also need the trial timestamps.csv
timestamps_file = bhv_directory / "sub-IR95-sess-3-trial_timestamps.csv"
import pandas as pd
timestamps = pd.read_csv(timestamps_file)

# final step is to get the first timepoint from the photodiode file
from intracranial_ephys_utils.load_data import get_event_times
event_folder = Path(f"{os.pardir}/data/{subject}/{session}/raw")
event_times, event_labels, global_start = get_event_times(event_folder, rescale=False)
microsec_sec = 10**-6
sec_microsec = 10**6
print(global_start)

# we're now poised to convert the trial timestamps into seconds from start of recording, and see if the relative times
# matches that of the differences from the events in the computer
trial_onsets = list(timestamps['img_onset'])
trial_onset_times = (trial_onsets-global_start)*10**-6
trial_offset_times = (list(timestamps['img_offset'])-global_start)*10**-6
import numpy as np
event_signal = np.zeros(ph_signal.shape)
for i in range(len(trial_onset_times)):
    event_signal[int(trial_onset_times[i] * sampling_rate):int(trial_offset_times[i] * sampling_rate)] = 1.
print('hip')