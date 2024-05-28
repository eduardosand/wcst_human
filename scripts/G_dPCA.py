import os
import numpy as np
from scipy.io import loadmat
from intracranial_ephys_utils.load_data import read_file, get_file_info
from C_su_raster_plots import get_trial_wise_times, get_spike_rate_curves, plot_neural_spike_trains
from behavior_analysis import process_wcst_behavior
from pathlib import Path
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib

def gaussian_smooth(spike_trains_sorted, sigma, step):
    """
    Take spike trains and convolve them with Gaussian kernel
    :param spike_trains_sorted:
    :param sigma:
    :param step:
    :return:
    """
    sigma = 0.05
    gx = np.arange(-4 * sigma, 4 * sigma, step)
    gaussian = np.exp(-(gx / sigma) ** 2 / 2)[:, np.newaxis]
    filtered_signals = scipy.signal.convolve(spike_trains_sorted.T, gaussian, mode='same').T
    return filtered_signals

# This script serves to perform a dPCA on the spiking data

# the first step towards that aim is to get my neural data and plot it

session = 'sess-3'
subject = 'IR95'
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
feedback_times = beh_timestamps['Feedback (seconds)']
onset_times = beh_timestamps['Onset (seconds)']


# global t-start
reader = read_file(ph_file_path)
reader.parse_header()
start_record = reader.global_t_start
number_spikes = []
curr_neuron = 0

su_data_dir = data_directory / "sorted/sort/final"
all_su_files = os.listdir(su_data_dir)

file = all_su_files[0]
max_rt = max(beh_data['response_time'])
# step 1. Load in one file, and comb through it for neurons
microwire_spikes = loadmat(su_data_dir / file)
neuron_counts = microwire_spikes['useNegative'][0].shape[0]
for neuron_ind in range(neuron_counts):
    su_cluster_num = microwire_spikes['useNegative'][0][neuron_ind]

    # Note that these timestamps are in microseconds, and according to machine clock
    microsec_sec_trans = 10**-6
    su_timestamps = np.array([[microwire_spikes['newTimestampsNegative'][0, i]*microsec_sec_trans-start_record] for i in
                                  range(microwire_spikes['newTimestampsNegative'].shape[1])
                                 if microwire_spikes['assignedNegative'][0, i] == su_cluster_num])

    # trial_wise_feedback_spikes = get_trial_wise_times(su_timestamps, feedback_times, beh_data, tmin=-1., tmax=1.5)
    # Plot response in spikes of this one neuron relative to each onset event
    tmin_onset = -1
    # tmax = max_rt+1.5
    tmax = 1.5
    trial_wise_feedback_spikes = get_trial_wise_times(su_timestamps, feedback_times, tmin=tmin_onset, tmax=tmax)

    sort_order = sorted(set(beh_data['correct']))
    if len(sort_order) == 3:
        color_dict = dict(zip(sort_order, ['red', 'green', 'blue']))
    else:
        color_dict = dict(zip(sort_order, ['purple', 'orange']))

    binsize = 0.01
    step = 0.01
    spike_trains_sorted, beh_conditions_sorted, change_indices = get_spike_rate_curves(trial_wise_feedback_spikes,
                                                                                       beh_data['correct'],
                                                                                       tmin=tmin_onset,
                                                                                       tmax=tmax, step=step,
                                                                                       binsize=binsize,
                                                                                       mode='single_trial')

    trial_time = np.arange(tmin_onset, tmax+step, step)
    sigma = 0.05
    filtered_signals = gaussian_smooth(spike_trains_sorted, sigma, step)
    fig, axs = plt.subplots(nrows=2)
    plot_neural_spike_trains(axs[0], trial_wise_feedback_spikes, beh_data['correct'], color_dict)
    axs[0].set_xlim([tmin_onset, tmax])
    for i in range(filtered_signals.shape[0]):
        axs[1].plot(trial_time, filtered_signals[i, :], color=color_dict[beh_conditions_sorted[i]])
    axs[0].set_title("Comparing raster plots and smoothed curves")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_xlim([tmin_onset, tmax])
    # axs[0].axis('off')
    plt.show()
    print(spike_trains_sorted)