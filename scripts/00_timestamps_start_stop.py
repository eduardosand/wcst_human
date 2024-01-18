from intracranial_ephys_utils.manual_process import reformat_event_labels, photodiode_check_viewer
from pathlib import Path
import os

# 1/18/2024
# This code is using some basic functionality to comb through the photodiode signal in an interactive data viewer,
# and mark down when the wcst experiment was, with the idea that when the photodiode is still, the data should largely
# have similar statistics, and make it easier to do photodiode processing (trial detection).

subject = 'IR87'
session = 'sess-4'
task = 'wcst'
sort_directory = Path(f"{os.pardir}/data/{subject}/{session}/")

data_directory = Path(f"{os.pardir}/data/{subject}/{session}/raw")

reformat_event_labels(subject, session, task, data_directory, sort_directory)
photodiode_check_viewer(subject, session, task, data_directory, sort_directory)
