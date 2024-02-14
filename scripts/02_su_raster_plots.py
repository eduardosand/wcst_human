import os
import numpy as np
from scipy.io import loadmat
from intracranial_ephys_utils.load_data import read_file
from behavior_analysis import process_wcst_behavior
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

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
        tmin = -1.
        tmax = 1.5
        linelength = 2
        # Plot response in spikes of this one neuron relative to each feedback event
        color_selection = ['purple', 'orange']
        for i in range(axs.shape[0]):
            trial_wise_feedback_spikes = []
            for trial_ind, feedback_time in enumerate(feedback_times):
                if trial_ind in list(beh_data.index):
                    # Plot the response in spikes of this one neuron to first feedback event
                    single_trial_spikes = (su_timestamps-feedback_time)
                    # select within bounds of analysis window (check for  both conditions)
                    single_trial_spikes = single_trial_spikes[((single_trial_spikes > tmin) * (single_trial_spikes < tmax))]
                    trial_wise_feedback_spikes.append([single_trial_spikes, beh_data['rule dimension'][trial_ind],
                                                       beh_data['correct'][trial_ind]])
            if i==0:
                # With the spikes in tow, we'll begin to sort them according rule dimension first
                sort_order = sorted(set(beh_data['rule dimension']))
                trial_wise_feedback_spikes.sort(key=lambda x: sort_order.index(x[1]))
                sorted_rule_dims = [trial_wise_feedback_spikes[trial_info][1] for trial_info in range(len(trial_wise_feedback_spikes))]
                sorted_feedback_spikes = [trial_wise_feedback_spikes[trial_info][0]
                                          for trial_info in range(len(trial_wise_feedback_spikes))]
                color_dict = dict(zip(sort_order, ['red', 'green', 'blue']))
                # spike_labels = sort_order
                axs[i, 0].eventplot(sorted_feedback_spikes, linelengths=linelength, colors=list(map(color_dict.get, sorted_rule_dims)))
                axs[i, 0].axvline(0, linestyle='--', c='black')
                axs[i, 0].set_title('Feedback-locked')
                tmin = -0.5
                tmax = 1.5
                # Plot response in spikes of this one neuron relative to each feedback event
                trial_wise_onset_spikes = []
                for trial_ind, onset_time in enumerate(onset_times):
                    if trial_ind in list(beh_data.index):
                        # Plot the response in spikes of this one neuron to first feedback event
                        single_trial_spikes = (su_timestamps - onset_time)
                        # select within bounds of analysis window (check for  both conditions)
                        single_trial_spikes = single_trial_spikes[((single_trial_spikes > tmin) * (single_trial_spikes < tmax))]
                        trial_wise_onset_spikes.append([single_trial_spikes, beh_data['rule dimension'][trial_ind]])

                trial_wise_onset_spikes.sort(key=lambda x: sort_order.index(x[1]))
                sorted_rule_dims = [trial_wise_onset_spikes[trial_info][1] for trial_info in
                                    range(len(trial_wise_onset_spikes))]
                sorted_onset_spikes = [trial_wise_onset_spikes[trial_info][0]
                                       for trial_info in range(len(trial_wise_onset_spikes))]
                spike_labels = sort_order
                axs[i, 1].eventplot(sorted_onset_spikes, linelengths=linelength,
                                    colors=list(map(color_dict.get, sorted_rule_dims)))

                axs[i, 1].axvline(0, linestyle='--', c='black')
                axs[i, 1].set_title("Onset-locked")
                # axs[1].eventplot(sorted_onset_spikes, linelengths=linelength, colors=list(map(color_dict.get, sorted_rule_dims)), label=sorted_rule_dims)
                # axs[1].legend(bbox_to_anchor=(0., 1.0, 1., .10), loc=3, ncol=3, mode="expand", borderaxespad=0.)
                # axs[1].legend()

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
                print('huh')
                # With the spikes in tow, we'll begin to sort them according rule dimension first
                sort_order = sorted(set(beh_data['correct']))
                trial_wise_feedback_spikes.sort(key=lambda x: sort_order.index(x[2]))
                sorted_feedback_cond = [trial_wise_feedback_spikes[trial_info][2] for trial_info in
                                    range(len(trial_wise_feedback_spikes))]
                sorted_feedback_spikes = [trial_wise_feedback_spikes[trial_info][0]
                                          for trial_info in range(len(trial_wise_feedback_spikes))]
                color_dict = dict(zip(sort_order, color_selection))
                # spike_labels = sort_order
                axs[i, 0].eventplot(sorted_feedback_spikes, linelengths=linelength,
                                    colors=list(map(color_dict.get, sorted_feedback_cond)))
                axs[i, 0].axvline(0, linestyle='--', c='black')
                axs[i, 0].set_xlabel("Time (s)")

                tmin = -0.5
                tmax = 1.5
                # Plot response in spikes of this one neuron relative to each feedback event
                trial_wise_onset_spikes = []
                for trial_ind, onset_time in enumerate(onset_times):
                    if trial_ind in list(beh_data.index):
                        # Plot the response in spikes of this one neuron to first feedback event
                        single_trial_spikes = (su_timestamps - onset_time)
                        # select within bounds of analysis window (check for  both conditions)
                        single_trial_spikes = single_trial_spikes[((single_trial_spikes > tmin) * (single_trial_spikes < tmax))]
                        trial_wise_onset_spikes.append([single_trial_spikes, beh_data['rule dimension'][trial_ind],
                                                        beh_data['correct'][trial_ind]])

                trial_wise_onset_spikes.sort(key=lambda x: sort_order.index(x[2]))
                sorted_feedback_cond = [trial_wise_onset_spikes[trial_info][2] for trial_info in
                                    range(len(trial_wise_onset_spikes))]
                sorted_onset_spikes = [trial_wise_onset_spikes[trial_info][0]
                                       for trial_info in range(len(trial_wise_onset_spikes))]
                axs[i, 1].eventplot(sorted_onset_spikes, linelengths=linelength,
                                    colors=list(map(color_dict.get, sorted_feedback_cond)))

                axs[i, 1].axvline(0, linestyle='--', c='black')
                axs[i, 1].set_xlabel("Time (s)")
                # axs[1].eventplot(sorted_onset_spikes, linelengths=linelength, colors=list(map(color_dict.get, sorted_rule_dims)), label=sorted_rule_dims)
                # axs[1].legend(bbox_to_anchor=(0., 1.0, 1., .10), loc=3, ncol=3, mode="expand", borderaxespad=0.)
                # axs[1].legend()

                # Create a summarized legend
                custom_legend = [
                    plt.Line2D([0], [0], color=color_dict[feedback_dim], lw=2, label=feedback_dim) for
                    feedback_dim in sort_order
                ]

                # Add the summarized legend to the plot
                # axs[1].legend(handles=custom_legend, loc='upper center')
                # Add the summarized legend to the right of the second subplot, but within the figure
                axs[i, 2].axis('off')  # Hide the axis
                axs[i, 2].legend(handles=custom_legend, loc='center', bbox_to_anchor=(1.05, 0.5),
                                 ncol=1)  # Adjust the position as needed
                plt.suptitle(f"Spike plot for cluster number {su_cluster_num}")
                plt.tight_layout()
            # axs[0].set_ylabel("Trial Number")


            # axs.eventplot(su_timestamps.T)
        plt.show()
print(poop)



neuron_count = 0
for file in all_su_files:
    data = loadmat(su_data_dir / file)
    # print(file)
    # print(data['useNegative'])
    # print(data['useNegative'].shape)
    neuron_count += data['useNegative'][0].shape[0]

# for each feedback time, start at -2.5 from that, and capture all spikes in a 100 ms window.
tmin = -2.5  # in seconds
tmax = 0.7  # in seconds

start = feedback_times+tmin
step = 0.1  # in seconds
overlap = 0.05
trial_time = np.linspace(tmin, tmax, int(abs(tmin-tmax)/overlap+1))

su_data = np.zeros((feedback_times.shape[0], neuron_count, trial_time.shape[0]-1))
print(all_su_files)
for file in all_su_files:
    data = loadmat(os.path.join(su_data_dir, file))

# useNegative - the clusters that were deemed single units
# newTimestampsNegative - timestamps for all detected spikes in the files
# assignedNegative - the cluster corresponding to each detected spike in newTimestampsNegative

    su_clusters = data['useNegative'][0]

    for cluster_ind, su_cluster_num in enumerate(su_clusters):
        # first axis is the spike number and second axis has two values, one of the timestamps and one of the cluster code
        su_timestamps = np.array([[data['newTimestampsNegative'][0, i], data['assignedNegative'][0, i]] for i in range(data['newTimestampsNegative'].shape[1])
                         if data['assignedNegative'][0, i] in [su_cluster_num]])
        print(su_timestamps.shape)
        number_spikes.append(su_timestamps.shape[0])
        # convert all spike_times to seconds
        su_time_convert = np.array([[su_timestamps[i, 1], su_timestamps[i, 0]*10**-6 - reader.global_t_start] for i in range(su_timestamps.shape[0])])

        # Now we have a few options
        # but let's keep it simple
        # we have all the event times, so instead of making a signal from scratch... We'll just go from -2.5 to 0.7 relative to feedback onset


        for trial_ind, feedback_onset in enumerate(feedback_times):
            for i in range(n_time):
                if start[trial_ind]+overlap*i+step > start[trial_ind]-tmin+tmax:
                    su_data[trial_ind, curr_neuron+cluster_ind, i] = np.sum(
                        (su_time_convert > start[trial_ind] + overlap * i) & (su_time_convert < start[trial_ind]-tmin+tmax))
                    # print(f'Data from {start[trial_ind] + overlap * i} to {start[trial_ind]-tmin+tmax}')
                else:
                    su_data[trial_ind, curr_neuron+cluster_ind, i] = np.sum((su_time_convert > start[trial_ind]+overlap*i) & (su_time_convert < start[trial_ind]+overlap*i+step))
                    # print(f'Data from {start[trial_ind]+overlap*i} to {start[trial_ind]+overlap*i+step}')
    curr_neuron += len(su_clusters)

print('stuff')

