from intracranial_ephys_utils.preprocess import save_small_dataset, broadband_seeg_processing
from intracranial_ephys_utils.load_data import read_task_ncs
import os
from pathlib import Path

test_subject = 'IR98'
test_session = 'sess-3'
test_task = 'wcst'
events_file_name = f'{test_subject}_{test_session}_{test_task}_events.csv'
events_path = Path(f'{os.pardir}/data/{test_subject}/{test_session}/{events_file_name}')

data_directory = Path(f"{os.pardir}/data/{test_subject}/{test_session}/raw")
low_pass = 1000
# Look in a microwire file
micro_file_path = "/mLHH1_0004.ncs"
micro_lfp_signal, sample_rate, _, macro_timestamps = read_task_ncs(data_directory, micro_file_path, task=test_task,
                                                       events_file=events_path)
processed_microwire, fs = broadband_seeg_processing(micro_lfp_signal, sample_rate, 0.1, low_pass)

# look in a macrocontact file
macro_file_path = "LHH1_0004.ncs"
macro_lfp_signal, sample_rate, _, micro_timestamps = read_task_ncs(data_directory, macro_file_path, task=test_task,
                                                       events_file=events_path)
processed_macrowire, fs = broadband_seeg_processing(macro_lfp_signal, sample_rate, 0.1, low_pass)

print(macro_timestamps)
print(len(macro_timestamps))

print(micro_timestamps)
print(len(micro_timestamps))
