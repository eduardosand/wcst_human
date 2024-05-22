from intracranial_ephys_utils.manual_process import (reformat_event_labels, photodiode_check_viewer,
                                                     diagnostic_time_series_plot)
from pathlib import Path
import os

# 1/18/2024
# This code is using some basic functionality to comb through the photodiode signal in an interactive data viewer,
# and mark down when the wcst experiment was, with the idea that when the photodiode is still, the data should largely
# have similar statistics, and make it easier to do photodiode processing (trial detection).

# subject = 'IR87'
# session = 'sess-4'
subject = 'IR95'
session = 'sess-2'
task = 'wcst'
sort_directory = Path(f"{os.pardir}/data/{subject}/{session}/")

data_directory = Path(f"{os.pardir}/data/{subject}/{session}/raw")

reformat_event_labels(subject, session, task, data_directory, sort_directory)


# def photodiode_check_viewer(subject, session, task, data_directory, annotations_directory, diagnostic=False,
#                             task_start=0.):
#     """
#     This script is a generalized dataviewer to look at a photodiode signal, and bring up the events
#     :param subject: The patient ID
#     :param session: Session of the experiment. Useful if patient completed more than one session of a task.
#     :param task: Which task
#     :param data_directory: Where the data lives
#     :param annotations_directory: Where we want to put the annotations of the data
#     :return:
#     """
#
#     all_files_list = os.listdir(data_directory)
#     # electrode_files = [file_path for file_path in all_files_list if (re.match('m.*ncs', file_path) and not
#     #                file_path.endswith(".nse"))]
#     ph_files = [file_path for file_path in all_files_list if file_path.endswith('.ncs') and
#                        file_path.startswith('photo1')]
#     assert len(ph_files) == 1
#     ph_filename = ph_files[0]
#     print(ph_filename)
#     ph_signal, sampling_rate, interp, timestamps = read_task_ncs(data_directory, ph_filename)
#     if diagnostic:
#         diagnostic_time_series_plot(ph_signal, sampling_rate, electrode_name='Photodiode')
#         start_time = input('Type start time (in seconds) if not the start of signal, else press enter: ')
#         end_time = input('Type end time (in seconds) if not the end of signal, else press enter: ')
#         if len(start_time) == 0:
#             start_time = 0
#         else:
#             start_time = int(start_time)
#         if len(end_time) == 0:
#             end_time = int(ph_signal.shape[0])
#         else:
#             end_time = int(end_time)
#
#         ph_signal = ph_signal[:int(sampling_rate * end_time)]
#         timestamps = timestamps[:int(sampling_rate * end_time)]
#         t_start = start_time
#
#         # next step to this is to add my thresholding for photodiode
#         # print(np.max(ph_signal))
#         # print(np.min(ph_signal))
#         ph_signal_bin = binarize_ph(ph_signal, sampling_rate)
#
#         event_signal = np.zeros(ph_signal.shape)
#         for i in range(len(trial_onset_times)):
#             event_signal[int(trial_onset_times[i]*sampling_rate):int(trial_offset_times[i]*sampling_rate)] = 1.
#         dataset = np.vstack([ph_signal, ph_signal_bin, event_signal]).T
#         print(event_signal)
#         labels = np.array([ph_filename, 'Photodiode Binarized', 'Ivan Events'])
#     else:
#         t_start = task_start
#         dataset = np.expand_dims(ph_signal, axis=1)
#         labels = np.expand_dims(np.array([ph_filename]), axis=1)
#
#     app = mkQApp()
#
#     # Create main window that can contain several viewers
#     win = MainViewer(debug=True, show_auto_scale=True)
#
#     # Create a viewer for signal
#     view1 = TraceViewer.from_numpy(dataset, sampling_rate, t_start, 'Photodiode', channel_names=labels)
#
#     view1.params['scale_mode'] = 'same_for_all'
#     view1.auto_scale()
#     win.add_view(view1)
#
#     possible_labels = [f'{task} duration']
#     file_path = annotations_directory / f'{subject}_{session}_{task}_events.csv'
#     source_epoch = CsvEpochSource(file_path, possible_labels)
#     # create a viewer for the encoder itself
#     view2 = EpochEncoder(source=source_epoch, name='Tagging events')
#     win.add_view(view2)
#     #
#     # view3 = EventList(source=source_epoch, name='events')
#     # win.add_view(view3)
#
#     # show main window and run Qapp
#     win.show()
#
#     app.exec()

#######
# There is an issue when photodiode signal is noisy, which is that this entire pipeline doesn't make a lot of sense
# anymore. In these cases, we'd like to process the photodiode signal if possible. However, if it's noisy there might
# some weird non-linearity(due to removing the photodiode from the screen in between sessions etc). So what we will do
# is first plot the entire photodiode signal at a glance to remove some of this, hoping that this doesn't interfere
# with annotations (as long as it's towards the end of the dataset it's fine).
photodiode_check_viewer(subject, session, task, data_directory, sort_directory, diagnostic=True)
