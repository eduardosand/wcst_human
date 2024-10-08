import os
import numpy as np
from scipy.io import loadmat
from intracranial_ephys_utils.load_data import read_file, get_file_info
from behavior_analysis import process_wcst_behavior
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy


def gaussian_smooth(spike_trains_sorted, sigma, step):
    """
    Take spike trains and convolve them with Gaussian kernel
    :param spike_trains_sorted:
    :param sigma:
    :param step:
    :return:
    """
    gx = np.arange(-4 * sigma, 4 * sigma, step)
    gaussian = np.exp(-(gx / sigma) ** 2 / 2)[:, np.newaxis]
    filtered_signals = scipy.signal.convolve(spike_trains_sorted.T, gaussian, mode='same').T
    return filtered_signals


def sort_spike_trains(spike_trains, beh_conditions):
    """
    Take spike_trains(time-locked) and behavioral conditions, and sort them for plots.
    :param spike_trains:
    :param beh_conditions:
    :return:
    """
    paired_list = list(zip(spike_trains, beh_conditions, range(len(spike_trains))))
    sorted_pairs = sorted(paired_list, key=lambda x: (x[1], x[2]))
    spike_trains_sorted = [sorted_pairs[i][0] for i in range(len(sorted_pairs))]
    beh_conditions_sorted = [sorted_pairs[i][1] for i in range(len(sorted_pairs))]
    # Get the indices where sorted conditions change, useful for computing PSTHs specific to conditions
    change_indices = np.where(np.array(beh_conditions_sorted)[:-1] != np.array(beh_conditions_sorted)[1:])[0]+1
    return spike_trains_sorted, beh_conditions_sorted, change_indices


def plot_neural_spike_trains(ax, spike_trains, beh_conditions, color_dict, line_length=2.5, line_width=1.):
    """
    Ideally this function takes in an axis and some spike train data, along with behavioral labels
    to generate a plot that colors the spike trains by condition and then plots them
    :param ax: Matplotlib axis object
    :param spike_trains: n_trials*n_timepoints. Should be pre-aligned
    :param beh_conditions: Labels for each trial
    :param color_dict: How the conditions should map to colors
    :param line_length: Length of line
    :return:
    """
    spike_trains_sorted, beh_conditions_sorted, change_indices = sort_spike_trains(spike_trains, beh_conditions)
    ax.eventplot(spike_trains_sorted, linelengths=line_length, linewidths=line_width,
                 colors=list(map(color_dict.get, beh_conditions_sorted)))
    ax.axvline(0, linestyle='--', c='black')


def get_spike_rate_curves(spike_trains, beh_conditions, tmin=-1., tmax=1.5, step=0.050, binsize=0.05, mode='average'):
    """
    Straightforward here. Trying to get full spike trains by either averaging in bins, or trial-wise spikes
    :param spike_trains:
    :param beh_conditions:
    :param tmin: (float) Minimum time to look relative to event lock time (predecided by spike_trains_sorted)
    :param tmax: (float) Max time in seconds to look relative to event lock time
    :param step: (float) How far forward we go when discretizing
    :param binsize: (float) Time interval to look for spikes for a given step. If smoothing after,
                            should be the same as step, otherwise is a form of smoothing itself
    :param mode:
    :return:
    """
    spike_trains_sorted, beh_conditions_sorted, change_indices_og = sort_spike_trains(spike_trains, beh_conditions)
    # next, count spikes in binsize ms bins, stepping by step ms
    trial_time = np.arange(tmin, tmax + step, step)
    if mode == 'average':
        num_conditions = len(np.unique(beh_conditions))
        change_indices = change_indices_og
    else:
        # basically just make change conditions, every single trial, to do single trial estimates (no averaging)
        num_conditions = len(beh_conditions)
        change_indices = np.arange(1, len(beh_conditions_sorted))
    spike_counts = np.zeros((num_conditions, trial_time.shape[0]))
    for cond_ind in range(num_conditions):
        for ind, timestep in enumerate(trial_time):
            if cond_ind == 0:
                all_times = np.concatenate(spike_trains_sorted[0:change_indices[0]])
                num_trials = change_indices[0]
            elif cond_ind == num_conditions - 1:
                all_times = np.concatenate(spike_trains_sorted[change_indices[cond_ind - 1]:])
                num_trials = len(spike_trains_sorted) - change_indices[cond_ind - 1]
            else:
                all_times = np.concatenate(spike_trains_sorted[change_indices[cond_ind - 1]:change_indices[cond_ind]])
                num_trials = change_indices[cond_ind] - change_indices[cond_ind - 1]
            spike_counts[cond_ind, ind] = np.shape(np.where((all_times < timestep + binsize / 2) &
                                                            (all_times >= timestep - binsize / 2)))[1] / num_trials
    return spike_counts, beh_conditions_sorted, change_indices_og


def plot_spike_rate_curves(ax, spike_trains, beh_conditions, color_dict, tmin=-1., tmax=1.5, gaussian_smooth_bool=False):
    """
    Plot spike counts over time per conditions, using 100 ms bins and stepping by 50 ms to help with smoothing.
    :param ax: Axis object
    :param spike_trains: list of lists, can't be turned into an array
    :param beh_conditions: list of labels. labels for each trial
    :param color_dict: Dictionary that tells what each behavioral conditions corresponds to for color labels
    :param tmin:
    :param tmax:
    :return:
    """
    num_conditions = len(np.unique(beh_conditions))
    spike_counts_sorted, beh_conditions_sorted, change_indices = get_spike_rate_curves(spike_trains, beh_conditions,
                                                                                       tmin, tmax)
    if gaussian_smooth_bool:
        sigma = 0.05
        step = 0.05
        spike_counts_sorted = gaussian_smooth(spike_counts_sorted, sigma, step)
    # next, count spikes in 100 ms bins, stepping by 50 ms
    step = 0.050
    trial_time = np.arange(tmin, tmax+step, step)
    for cond_ind in range(num_conditions):
        if cond_ind == 0:
            label_val = beh_conditions_sorted[0]
            ax.plot(trial_time, spike_counts_sorted[cond_ind, :], color=color_dict[label_val], label=label_val)
        elif cond_ind == num_conditions-1:
            label_val = beh_conditions_sorted[-1]
            ax.plot(trial_time, spike_counts_sorted[cond_ind, :], color=color_dict[label_val], label=label_val)
        else:
            label_val = beh_conditions_sorted[change_indices[cond_ind-1]]
            ax.plot(trial_time, spike_counts_sorted[cond_ind, :], color=color_dict[label_val], label=label_val)
    ax.axvline(0, linestyle='--', c='black')
    ymax = max(np.max(spike_counts_sorted[:, :]), 0.1)
    ax.set_ylim([0, ymax])


def get_trial_wise_spike_times(su_timestamps, trial_times, tmin=-0.5, tmax=1.5):
    """
    This function converts spike times in original timing to trial locked spike times using both
    timestamps, and behavioral data to check for valid trial.
    More specifically, this function is converting the timestamps of each spike relative to all relevant trial times
    Then for a single trial, we only grab those spikes that happened within a window. The caveat to this is that if we
    make our tmin or tmax too large(absolutely), then we run the risk of using the same spikes across multiple times
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
        # if trial_ind in list(beh_data.index):
        # Plot the response in spikes of this one neuron to first feedback event
        single_trial_spikes = (su_timestamps - trial_time)
        # select within bounds of analysis window (check for  both conditions)
        single_trial_spikes = single_trial_spikes[
                    ((single_trial_spikes > tmin) * (single_trial_spikes < tmax))]
        trial_wise_spikes.append(single_trial_spikes)
    return trial_wise_spikes

# session = 'sess-4'
# subject = 'IR87'

# TODOlater Turn the below into a reasonable function, for now just make it the main
def main():
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

    matplotlib.rcParams['font.size'] = 12.0

    # global t-start
    reader = read_file(ph_file_path)
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

            fig, axs = plt.subplots(4, 3, sharey='row', sharex='col', figsize=(6, 6),
                                    gridspec_kw={'width_ratios': [1, 1, 0.4]})
            tmax = 2.5
            trial_wise_feedback_spikes = get_trial_wise_spike_times(su_timestamps, feedback_times, tmin=-1., tmax=tmax)
            # Plot response in spikes of this one neuron relative to each onset event
            tmin_onset = -0.5
            trial_wise_onset_spikes = get_trial_wise_spike_times(su_timestamps, onset_times, tmin=tmin_onset, tmax=tmax)
            gaussian_smooth_bool = True
            label_fontsize = 18
            for i in range(int(axs.shape[0]/2)):
                axs[i * 2, 0].set_ylabel("Spiking", fontsize=label_fontsize)
                if gaussian_smooth_bool:
                    axs[i * 2 + 1, 0].set_ylabel('Firing \n Rate', fontsize=label_fontsize)
                else:
                    axs[i * 2 + 1, 0].set_ylabel('Spike \n Counts', fontsize=label_fontsize)
                if i == 0:
                    # With the spikes in tow, we'll begin to sort them according rule dimension first
                    sort_order = sorted(set(beh_data['rule dimension']))
                    if len(sort_order) == 3:
                        color_dict = dict(zip(sort_order, ['red', 'green', 'blue']))
                    else:
                        color_dict = dict(zip(sort_order, ['purple', 'orange']))
                    plot_neural_spike_trains(axs[i, 0], trial_wise_feedback_spikes, beh_data['rule dimension'], color_dict)

                    plot_spike_rate_curves(axs[i+1, 0], trial_wise_feedback_spikes, beh_data['rule dimension'],
                                           color_dict, tmin=-1, tmax=tmax, gaussian_smooth_bool=True)
                    axs[i, 0].set_title('Feedback \n locked', fontsize=label_fontsize)
                    plot_neural_spike_trains(axs[i, 1], trial_wise_onset_spikes, beh_data['rule dimension'], color_dict)
                    plot_spike_rate_curves(axs[i+1, 1], trial_wise_onset_spikes, beh_data['rule dimension'], color_dict,
                                           tmin=tmin_onset, tmax=tmax, gaussian_smooth_bool=True)
                    axs[i, 1].set_title("Onset \n locked", fontsize=label_fontsize)
                    # Create a summarized legend,
                    custom_legend = [
                        plt.Line2D([0], [0], color=color_dict[rule], lw=2, label=rule) for rule in sort_order
                    ]

                    # Add the summarized legend to the plot
                    # axs[1].legend(handles=custom_legend, loc='upper center')
                    # Add the summarized legend to the right of the second subplot, but within the figure

                    axs[i+1, 2].axis('off')  # Hide the axis
                    axs[i, 2].axis('off')  # Hide the axis
                    axs[i, 2].legend(handles=custom_legend, loc='center', bbox_to_anchor=(1.05, 0.5),
                                     ncol=1, fontsize=15)  # Adjust the position as needed
                    plt.suptitle(f"Single Unit {su_cluster_num} Activity", fontsize=30)
                    plt.tight_layout()
                else:
                    sort_order = sorted(set(beh_data['correct']))
                    if len(sort_order) == 3:
                        color_dict = dict(zip(sort_order, ['red', 'green', 'blue']))
                    else:
                        color_dict = dict(zip(sort_order, ['purple', 'orange']))
                    plot_neural_spike_trains(axs[i*2, 0], trial_wise_feedback_spikes, beh_data['correct'], color_dict)
                    plot_spike_rate_curves(axs[i*2+1, 0], trial_wise_feedback_spikes, beh_data['correct'], color_dict,
                                           tmin=-1, tmax=tmax, gaussian_smooth_bool=True)
                    plot_neural_spike_trains(axs[i*2, 1], trial_wise_onset_spikes, beh_data['correct'], color_dict)
                    plot_spike_rate_curves(axs[i*2+1, 1], trial_wise_onset_spikes, beh_data['correct'], color_dict,
                                           tmin=tmin_onset, tmax=tmax, gaussian_smooth_bool=True)
                    axs[i*2+1, 0].set_xlabel("Time (s)", fontsize=label_fontsize)
                    axs[i*2+1, 1].set_xlabel("Time (s)", fontsize=label_fontsize)
                    # Create a summarized legend
                    custom_legend = [
                        plt.Line2D([0], [0], color=color_dict[feedback_dim], lw=2, label=feedback_dim) for
                        feedback_dim in sort_order
                    ]

                    axs[i*2+1, 2].axis('off')  # Hide the axis
                    axs[i*2, 2].axis('off')  # Hide the axis
                    axs[i*2, 2].legend(handles=custom_legend, loc='center', bbox_to_anchor=(1.05, 0.5),
                                       ncol=1, fontsize=15)  # Adjust the position as needed
                    plt.tight_layout()
            plt.savefig(f"{os.pardir}/results/{subject}_{session}_cluster_{su_cluster_num}.svg")
            plt.show()


if __name__ == "__main__":
    main()
