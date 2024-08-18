from intracranial_ephys_utils.preprocess import broadband_seeg_processing
from intracranial_ephys_utils.load_data import read_task_ncs, missing_samples_check
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
micro_file_path = "mLHH1_0004.ncs"
micro_lfp_signal, micro_sample_rate, micro_interp, micro_timestamps = read_task_ncs(data_directory, micro_file_path, task=test_task,
                                                       events_file=events_path)
processed_microwire, fs = broadband_seeg_processing(micro_lfp_signal, micro_sample_rate, 0.1, low_pass)
micro_length = (micro_timestamps[-1]-micro_timestamps[0])
# look in a macrocontact file
macro_file_path = "LHH1_0004.ncs"
macro_lfp_signal, macro_sample_rate, macro_interp, macro_timestamps = read_task_ncs(data_directory, macro_file_path, task=test_task,
                                                       events_file=events_path)
processed_macrowire, fs = broadband_seeg_processing(macro_lfp_signal, macro_sample_rate, 0.1, low_pass)
macro_length = (macro_timestamps[-1]-macro_timestamps[0])
print(macro_timestamps)
print(len(macro_timestamps))

print(micro_timestamps)
print(len(micro_timestamps))
print('done')