"""
Temporary. Here we'll just test the extent to which our scripts still work on
raw behavior data
"""

from behavior_analysis import process_wcst_behavior
from behavior_enumeration import enumerate_cond_sess
from pathlib import Path
import os
import re

subject = 'IR95'
session = '1'
file_name_begin = f'{subject}_wcst'
extension = '.csv'
running_avg = 5
data_directory = Path(f"{os.pardir}/data/{subject}/sess-{session}/behavior")
file_names = [file_name for file_name in os.listdir(data_directory) if (file_name_begin in file_name and extension in file_name) ]
if len(file_names) == 1:
    file_path = data_directory / file_names[0]
else:
    raise NotImplementedError

beh_data, rule_shifts_ind, _ = process_wcst_behavior(file_path, running_avg=running_avg)

counts_dict = enumerate_cond_sess(beh_data)

print(counts_dict)
