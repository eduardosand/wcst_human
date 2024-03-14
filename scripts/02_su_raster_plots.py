import os
import numpy as np
from scipy.io import loadmat
from intracranial_ephys_utils.load_data import read_file
from behavior_analysis import process_wcst_behavior
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def plot_neural_spike_trains(ax, spike_trains, beh_conditions, color_dict, tmin=-1., tmax=1.5):
    """
    Ideally this function takes in an axis and some spike train data, along with behavioral labels
    to generate a plot that colors the spike trains by condition and then plots them
    :param ax: Matplotlib axis object
    :param spike_trains: n_trials*n_timepoints. Should be pre-aligned
    :param beh_conditions: Labels for each trial
    :param color_dict: How the conditions should map to colors
    :return:
    """
    num_conditions = len(np.unique(beh_conditions))
    paired_list = list(zip(spike_trains, beh_conditions, range(len(spike_trains))))
    sorted_pairs = sorted(paired_list, key=lambda x: (x[1], x[2]))
    spike_trains_sorted = [sorted_pairs[i][0] for i in range(len(sorted_pairs))]
    beh_conditions_sorted = [sorted_pairs[i][1] for i in range(len(sorted_pairs))]
    # Get the indices where sorted conditions change, useful for computing psths specific to conditions
    change_indices = np.where(np.array(beh_conditions_sorted)[:-1] != np.array(beh_conditions_sorted)[1:])[0]+1

    # next, count spikes in 100 ms bins, stepping by 50 ms
    step = 0.050
    trial_time = np.arange(tmin, tmax+step, step)
    spike_counts = np.zeros((num_conditions, trial_time.shape[0]))
    binsize = 0.1
    for cond_ind in range(num_conditions):
        for ind, timestep in enumerate(trial_time):
            if cond_ind == 0:
                all_times = np.concatenate(spike_trains_sorted[0:change_indices[0]])
            elif cond_ind == num_conditions-1:
                all_times = np.concatenate(spike_trains_sorted[change_indices[cond_ind-1]:])
            else:
                all_times = np.concatenate(spike_trains_sorted[change_indices[cond_ind-1]:change_indices[cond_ind]])
            print(timestep)
            print(timestep+binsize)
            print(np.shape(np.where((all_times < timestep+binsize) & (all_times >= timestep))))
            spike_counts[cond_ind, ind] = np.shape(np.where((all_times < timestep+binsize) & (all_times >= timestep)))[1]
            print(spike_counts[0, ind])


    ax.eventplot(spike_trains_sorted, linelengths=linelength,
                        colors=list(map(color_dict.get, beh_conditions_sorted)))

    ax.axvline(0, linestyle='--', c='black')
    ax.set_xlabel("Time (s)")


def plot_spike_rate_curves(ax, spike_trains, beh_conditions, color_dict, tmin=-1., tmax=1.5):
    """
    Plot spike counts over time per conditions, using 100 ms bins and stepping by 50 ms to help with smoothing.
    :param ax:
    :param spike_trains:
    :param beh_conditions:
    :param color_dict:
    :param tmin:
    :param tmax:
    :return:
    """
    num_conditions = len(np.unique(beh_conditions))
    paired_list = list(zip(spike_trains, beh_conditions, range(len(spike_trains))))
    sorted_pairs = sorted(paired_list, key=lambda x: (x[1], x[2]))
    spike_trains_sorted = [sorted_pairs[i][0] for i in range(len(sorted_pairs))]
    beh_conditions_sorted = [sorted_pairs[i][1] for i in range(len(sorted_pairs))]
    # Get the indices where sorted conditions change, useful for computing psths specific to conditions
    change_indices = np.where(np.array(beh_conditions_sorted)[:-1] != np.array(beh_conditions_sorted)[1:])[0]+1

    # next, count spikes in 100 ms bins, stepping by 50 ms
    step = 0.050
    trial_time = np.arange(tmin, tmax+step, step)
    spike_counts = np.zeros((num_conditions, trial_time.shape[0]))
    binsize = 0.1
    for cond_ind in range(num_conditions):
        for ind, timestep in enumerate(trial_time):
            if cond_ind == 0:
                all_times = np.concatenate(spike_trains_sorted[0:change_indices[0]])
            elif cond_ind == num_conditions-1:
                all_times = np.concatenate(spike_trains_sorted[change_indices[cond_ind-1]:])
            else:
                all_times = np.concatenate(spike_trains_sorted[change_indices[cond_ind-1]:change_indices[cond_ind]])
            print(timestep)
            print(timestep+binsize)
            print(np.shape(np.where((all_times < timestep+binsize) & (all_times >= timestep))))
            spike_counts[cond_ind, ind] = np.shape(np.where((all_times < timestep+binsize) & (all_times >= timestep)))[1]
            print(spike_counts[0, ind])
    for cond_ind in range(num_conditions):
        if cond_ind == 0:
            label_val = beh_conditions_sorted[0]
            ax.plot(trial_time, spike_counts[cond_ind, :], color=color_dict[label_val], label=label_val)
        elif cond_ind == num_conditions-1:
            label_val = beh_conditions_sorted[-1]
            ax.plot(trial_time, spike_counts[cond_ind, :], color=color_dict[label_val], label=label_val)
        else:
            label_val = beh_conditions_sorted[change_indices[cond_ind-1]]
            ax.plot(trial_time, spike_counts[cond_ind, :], color=color_dict[label_val], label=label_val)

    # ax.axvline(0, linestyle='--', c='black')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel('Spike Counts')

def get_trial_wise_times(su_timestamps, trial_times, beh_data, tmin=-0.5, tmax=1.5):
    """
    This function converts spike times in original timing to trial locked spike times using both
    timestamps, and behavioral data to check for valid trial.
    :param su_timestamps:
    :param trial_times:
    :param beh_data:
    :param tmin: defines how much before trial locked time to look for spikes
    :param tmax: defines how much after trial locked time to look for spikes
    :return:
    """
    # Plot response in spikes of this one neuron relative to each feedback event
    trial_wise_spikes = []
    for trial_ind, trial_time in enumerate(trial_times):
        if trial_ind in list(beh_data.index):
            # Plot the response in spikes of this one neuron to first feedback event
            single_trial_spikes = (su_timestamps - trial_time)
            # select within bounds of analysis window (check for  both conditions)
            single_trial_spikes = single_trial_spikes[
                    ((single_trial_spikes > tmin) * (single_trial_spikes < tmax))]
            trial_wise_spikes.append(single_trial_spikes)
    return trial_wise_spikes


session = 'sess-4'
subject = 'IR87'
data_directory = Path(f"{os.pardir}/data/{subject}/{session}/")
ph_file_path = Path("raw/Events.nev")
beh_directory = data_directory / "behavior"
running_avg = 5
beh_data, rule_shifts_ind, _ = process_wcst_behavior(beh_directory / f"sub-{subject}-{session}-beh.csv",
                                                     running_avg=running_avg)
beh_data.set_index(['trial'], inplace=True)
timestamps_file = "sub-IR87-sess-4-ph_timestamps.csv"
beh_timestamps = pd.read_csv(beh_directory / timestamps_file)
feedback_times = beh_timestamps['Feedback (seconds)']
onset_times = beh_timestamps['Onset (seconds)']

matplotlib.rcParams['font.size'] = 12.0

# global t-start
reader = read_file(data_directory / ph_file_path)
reader.parse_header()
start_record = reader.global_t_start
number_spikes = []
curr_neuron = 0

su_data_dir = data_directory / "sorted/sort/final"
all_su_files = os.listdir(su_data_dir)
for file in all_su_files:
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

        fig, axs = plt.subplots(2, 3, sharey=True, figsize=(6, 6), gridspec_kw={'width_ratios': [1, 1, 0.4]})
        linelength = 2
        trial_wise_feedback_spikes = get_trial_wise_times(su_timestamps, feedback_times, beh_data, tmin=-1., tmax=1.5)
        # Plot response in spikes of this one neuron relative to each onset event
        trial_wise_onset_spikes = get_trial_wise_times(su_timestamps, onset_times, beh_data, tmin=-0.5, tmax=1.5)
        for i in range(axs.shape[0]):
            if i == 0:
                # With the spikes in tow, we'll begin to sort them according rule dimension first
                sort_order = sorted(set(beh_data['rule dimension']))
                if len(sort_order) == 3:
                    color_dict = dict(zip(sort_order, ['red', 'green', 'blue']))
                else:
                    color_dict = dict(zip(sort_order, ['purple', 'orange']))
                # plot_neural_spike_trains(axs[i, 0], trial_wise_feedback_spikes, beh_data['rule dimension'], color_dict)

                plot_spike_rate_curves(axs[i, 0], trial_wise_feedback_spikes, beh_data['rule dimension'], color_dict)
                axs[i, 0].set_title('Feedback-locked')
                # Plot response in spikes of this one neuron relative to each feedback event

                plot_neural_spike_trains(axs[i, 1], trial_wise_onset_spikes, beh_data['rule dimension'], color_dict)
                axs[i, 1].set_title("Onset-locked")

                # Create a summarized legend
                custom_legend = [
                    plt.Line2D([0], [0], color=color_dict[rule], lw=2, label=rule) for rule in sort_order
                ]

                # Add the summarized legend to the plot
                # axs[1].legend(handles=custom_legend, loc='upper center')
                # Add the summarized legend to the right of the second subplot, but within the figure
                axs[i, 2].axis('off')  # Hide the axis
                axs[i, 2].legend(handles=custom_legend, loc='center', bbox_to_anchor=(1.05, 0.5),
                                 ncol=1)  # Adjust the position as needed
                plt.suptitle(f"Spike plot for cluster number {su_cluster_num}")
                plt.tight_layout()
            else:
                sort_order = sorted(set(beh_data['correct']))
                if len(sort_order) == 3:
                    color_dict = dict(zip(sort_order, ['red', 'green', 'blue']))
                else:
                    color_dict = dict(zip(sort_order, ['purple', 'orange']))
                plot_neural_spike_trains(axs[i, 0], trial_wise_feedback_spikes, beh_data['correct'], color_dict)
                plot_neural_spike_trains(axs[i, 1], trial_wise_onset_spikes, beh_data['correct'], color_dict)

                # Create a summarized legend
                custom_legend = [
                    plt.Line2D([0], [0], color=color_dict[feedback_dim], lw=2, label=feedback_dim) for
                    feedback_dim in sort_order
                ]

                axs[i, 2].axis('off')  # Hide the axis
                axs[i, 2].legend(handles=custom_legend, loc='center', bbox_to_anchor=(1.05, 0.5),
                                 ncol=1)  # Adjust the position as needed
                plt.suptitle(f"Spike plot for cluster number {su_cluster_num}")
                plt.tight_layout()
            # axs[0].set_ylabel("Trial Number")


            # axs.eventplot(su_timestamps.T)
        plt.show()



