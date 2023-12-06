"""
Temporary. Here we'll just test the extent to which our scripts still work on
raw behavior data
"""

from behavior_analysis import process_wcst_behavior
from behavior_enumeration import enumerate_cond_sess
from pathlib import Path
import os


subject = 'IR92'
session = '1'
file_name = f'{subject}_wcst6_2020_Jan_08_1338.csv'
running_avg = 5
data_directory = Path(f"{os.pardir}/data/{subject}/sess-{session}/behavior")
file_path = data_directory / file_name

beh_data, rule_shifts_ind, _ = process_wcst_behavior(file_path, running_avg=running_avg)

counts_dict = enumerate_cond_sess(beh_data)

print(counts_dict)
