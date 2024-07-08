from intracranial_ephys_utils.preprocess import save_small_dataset
import os
from pathlib import Path

test_subject = 'IR95'
test_session = 'sess-3'
test_task = 'wcst'
events_file_name = f'{test_subject}_{test_session}_{test_task}_events.csv'
events_path = Path(f'{os.pardir}/data/{test_subject}/{test_session}/{events_file_name}')
save_small_dataset(test_subject, test_session, task_name=test_task, events_file=events_path)