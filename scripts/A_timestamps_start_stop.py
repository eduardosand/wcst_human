from intracranial_ephys_utils.manual_process import (reformat_event_labels, photodiode_check_viewer,
                                                     diagnostic_time_series_plot)
from pathlib import Path
import os

# 1/18/2024
# This code is using some basic functionality to comb through the photodiode signal in an interactive data viewer,
# and mark down when the wcst experiment was, with the idea that when the photodiode is still, the data should largely
# have similar statistics, and make it easier to do photodiode processing (trial detection).

subject = 'IR98'
session = 'sess-1'
# subject = 'IR95'
# session = 'sess-1'
task = 'wcst'
sort_directory = Path(f"{os.pardir}/data/{subject}/{session}/")

data_directory = Path(f"{os.pardir}/data/{subject}/{session}/raw")

reformat_event_labels(subject, session, task, data_directory, sort_directory)

#######
# There is an issue when photodiode signal is noisy, which is that this entire pipeline doesn't make a lot of sense
# anymore. In these cases, we'd like to process the photodiode signal if possible. However, if it's noisy there might
# some weird non-linearity(due to removing the photodiode from the screen in between sessions etc). So what we will do
# is first plot the entire photodiode signal at a glance to remove some of this, hoping that this doesn't interfere
# with annotations (as long as it's towards the end of the dataset it's fine).
photodiode_check_viewer(subject, session, task, data_directory, sort_directory, diagnostic=False)
# photodiode_check_viewer(subject, session, task, data_directory, sort_directory, diagnostic=True)
