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
import mne
from dPCA import dPCA
from sklearn.decomposition import PCA


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


def featurize(epochs_object, feature, norm=False):
    """
    This functions assumes mne epochs_object is provided which is set up as
    n_epochs(trials) X n_electrodes(observations) X n_timepoints(observations in time)
    :param epochs_object: MNE epochs object. The data in this object should be n_trials X n_electrodes X n_timepoints
    :param feature:
    :param norm:
    :return:
    """
    epochs_dataset = epochs_object.get_data(copy=True)
    n_epochs, n_electrodes, n_timepoints = epochs_dataset.shape
    features = set(list(feature))

    feature_dict = {}
    rule_dim_key_values = [(value, ind) for ind, value in enumerate(features)]
    feature_dict.update(rule_dim_key_values)
    # rule_split = [list(rule[~np.isnan(rt)]).count(i) for i in set(list(rule[~np.isnan(rt)]))]
    feature_split = [list(feature).count(i) for i in features]
    # Here, we're making a better epochs object that is also factored by condition
    # n_trials = min(feature_split)
    organized_data = np.zeros((max(feature_split), n_electrodes, len(features), n_timepoints))
    # organized_data = np.zeros((n_trials, n_electrodes, len(rule_dimensions), n_timepoints))
    organized_data[organized_data == 0] = np.nan
    for electrode in range(n_electrodes):
        counts_dict = {}
        key_values = [(i, 0) for i in feature]
        counts_dict.update(key_values)
        for epoch in range(n_epochs):
            curr_feature = list(feature)[epoch]
            # print(curr_feature)
            # if counts_dict[curr_feature] >= min(feature_split):
                # print(curr_feature)
                # break
                # continue
            # else:
                # print(epochs_dataset[epoch, electrode, :])
            curr_count = counts_dict[curr_feature]
                    # print(curr_count)
            organized_data[curr_count, electrode, feature_dict[curr_feature], :] = epochs_dataset[epoch,
                                                                                                  electrode, :]
            counts_dict[curr_feature] += 1
                # print(counts_dict)
    # Center our data within electrodes
    # organized_data -= np.nanmean(organized_data.reshape((n_trials, -1)), 1)[:, None, None]
    # trial-average-data
    if norm:
        organized_data_mean = np.nanmean(organized_data, axis=0)
        organized_data_mean -= np.nanmean(organized_data_mean.reshape((n_electrodes, -1)), 1)[:, None, None]
        organized_data_mean /= np.nanstd(organized_data_mean.reshape((n_electrodes, -1)), 1)[:, None, None]
    else:
        organized_data_mean = np.nanmean(organized_data, axis=0)
        organized_data_mean -= np.nanmean(organized_data_mean.reshape((n_electrodes, -1)), 1)[:, None, None]
    return organized_data_mean, organized_data, feature_dict


def plot_signal_avg(organized_data_mean, subject, session, trial_time,
                    labels=[], extra_string=''):
    """
    This is a sanity function. After featurizing the data and computing signal average for each condition for each electrode,
    this will plot it. Note this function is a work in progress, and odds are that the False feedback_locked mode should
    be looked at and fixed.
    :param organized_data_mean: (array) we expect to be of the shape n_electrodes/neurons X n_conditions X n_timepoints
    :param subject: (string) subject identifier
    :param session: (session) subject identifier
    :param trial_time: (array) The timepoints of what each sample in organized data mean corresponds to
    :param feedback_locked:
    :return:
    """
    n_electrodes, n_cond, n_time = organized_data_mean.shape
    bin = 0.5
    min_multiple = np.min(trial_time) // 0.5
    # pad = 0.2
    time_ticks = np.arange(min_multiple*bin, np.max(trial_time), step=0.5)
    time_tick_labels = time_ticks
    # time_ticks = np.linspace(0, (abs(baseline_start)+tmax-pad)*fs, 7)
    # time_tick_labels = np.linspace(baseline_start, tmax-pad, 7)
    # x_zero = np.where(time_tick_labels == 0.)
    # time_tick_labels = np.insert(time_tick_labels, 0, baseline_start)
    time_tick_labels = [f'{i:.1f}' for i in time_tick_labels]

    ncols = 2
    n_electrodes, n_cond, n_timepoints = organized_data_mean.shape
    if n_electrodes % 2 == 1:
        n_plots = n_electrodes + 1
    else:
        n_plots = n_electrodes
    fig, ax = plt.subplots(int(n_plots/ncols), ncols)
    if len(labels) == 0:
        labels = np.arange(n_cond)

    if len(labels) == 3:
        color_dict = dict(zip(labels, ['red', 'green', 'blue']))
    elif len(labels) == 2:
        color_dict = dict(zip(labels, ['purple', 'orange']))
    print(color_dict)
    custom_legend = [
        plt.Line2D([0], [0], color=color_dict[label], lw=2, label=label) for label in labels
        ]
    for ind, ax_curr in enumerate(ax.flatten()):
        if ind >= n_electrodes:
            ax.flatten()[ind-2].set_xlabel('Time (s)')
            ax.flatten()[ind-2].set_xticks(time_ticks, time_tick_labels)
            continue
        for cond in range(n_cond):
            ax_curr.plot(trial_time, organized_data_mean[ind, cond], label=labels[cond], color=color_dict[labels[cond]])

            ax_curr.axvline(0, linestyle='--', c='black')
        if ind in np.arange(n_plots-ncols, n_plots):
            ax_curr.set_xlabel('Time (s)')
            ax_curr.set_xticks(time_ticks, time_tick_labels)
                # Create a summarized legend
    ax.flatten()[-1].axis('off')  # Hide the axis
    ax.flatten()[-1].axis('off')
    ax.flatten()[-1].legend(handles=custom_legend, loc='center', bbox_to_anchor=(1.05, 0.5), ncol=1)  # Adjust the position as needed
    plt.suptitle(f'Mean Activity of SU \n {subject} - session {session} \n {extra_string}')
    # plt.legend()
    # plt.tight_layout()
    plt.show()


def plot_dPCA_components(dpca, Z, trial_time, features, subject, session, suptitle,
                         feature_names=[]):
    """
    Plotting the dPCA components, and the features they correspond to.
    :param dpca: dPCA object already fitted
    :param Z:
    :param trial_time: (array) This is the x-axis for the plots, should correspond to event-locked time
    :param features:
    :param subject:
    :param session:
    :param suptitle: (string) Title of the plot
    :param feature_names:
    :return:
    """
    # time_ticks, time_tick_labels = get_ticks(feedback_locked, fs)

    # time = np.arange(n_timepoints)

    plt.figure(figsize=(16, 7))
    plt.subplot(131)

    # The underlying thing here is that Z should have the shape n_components * n_features_1 * ... * n_features_n * n_timepoints
    # One hack is to just check for the length of the shape
    n_features = len(Z['t'].shape) - 2
    if n_features > 1:

        n_features_1, n_features_2 = features.shape
        inv_feature_one_dict = {}
        inv_feature_key_values = [(ind, val[:-1]) for ind, val in enumerate(features[:, 0])]
        inv_feature_one_dict.update(inv_feature_key_values)

        inv_feature_two_dict = {}
        inv_feature_key_values = [(ind, val[-1]) for ind, val in enumerate(features[0, :])]
        inv_feature_two_dict.update(inv_feature_key_values)
        for f in range(n_features_1):
            for r in range(n_features_2):
                plt.plot(trial_time, Z['t'][0, f, r], label=inv_feature_one_dict[f]+inv_feature_one_dict[r])
    else:
        inv_rule_dim_dict = {}
        inv_feature_key_values = [(ind, val) for ind, val in enumerate(features)]
        inv_rule_dim_dict.update(inv_feature_key_values)
        for s in range(len(features)):
            plt.plot(trial_time, Z['t'][0, s], label=inv_rule_dim_dict[s])
        plt.axvline(0, linestyle='--', c='black')

    plt.title(f'1st time component \n explained variance: {dpca.explained_variance_ratio_["t"][0]:.2%}')
    # plt.xticks(time_ticks, time_tick_labels)
    plt.xlabel('Time (s)')
    plt.legend()
    plt.subplot(132)
    if n_features > 1:

        n_features_1, n_features_2 = features.shape
        inv_feature_one_dict = {}
        inv_feature_key_values = [(ind, val[:-1]) for ind, val in enumerate(features[:, 0])]
        inv_feature_one_dict.update(inv_feature_key_values)

        inv_feature_two_dict = {}
        inv_feature_key_values = [(ind, val[-1]) for ind, val in enumerate(features[0, :])]
        inv_feature_two_dict.update(inv_feature_key_values)
        for f in range(n_features_1):
            for r in range(n_features_2):
                plt.plot(trial_time, Z['f'][0, f, r], label=inv_feature_one_dict[f]+inv_feature_two_dict[r])
        plt.axvline(0, linestyle='--', c='black')
        plt.title(f'1st feedback component \n explained variance: {dpca.explained_variance_ratio_["f"][0]:.2%}')
    else:
        for s in range(len(features)):
            plt.plot(trial_time, Z['s'][0, s], label=inv_rule_dim_dict[s])
        plt.axvline(0, linestyle='--', c='black')
        if len(feature_names) == 0:
            plt.title(f'1st rule component \n explained variance: {dpca.explained_variance_ratio_["s"][0]:.2%}')
        else:
            plt.title(f'1st {feature_names[0]} component \n explained variance: {dpca.explained_variance_ratio_["s"][0]:.2%}')

    # plt.xticks(time_ticks, time_tick_labels)
    plt.xlabel('Time (s)')
    # pca.explained_variance_ratio_
    plt.legend()
    plt.subplot(133)

    if n_features > 1:

        n_features_1, n_features_2 = features.shape
        inv_feature_one_dict = {}
        inv_feature_key_values = [(ind, val[:-1]) for ind, val in enumerate(features[:, 0])]
        inv_feature_one_dict.update(inv_feature_key_values)

        inv_feature_two_dict = {}
        inv_feature_key_values = [(ind, val[-1]) for ind, val in enumerate(features[0, :])]
        inv_feature_two_dict.update(inv_feature_key_values)
        for f in range(n_features_1):
            for r in range(n_features_2):
                plt.plot(trial_time, Z['r'][0, f, r], label=inv_feature_one_dict[f] + inv_feature_two_dict[r])
        plt.title(f'1st rule component \n explained variance: {dpca.explained_variance_ratio_["r"][0]:.2%}')
    else:
        for s in range(len(features)):
            plt.plot(trial_time, Z['st'][0, s])
        plt.axvline(0, linestyle='--', c='black')
        plt.title(f'1st mixed component \n explained variance: {dpca.explained_variance_ratio_["st"][0]:.2%}')

    # plt.xticks(time_ticks, time_tick_labels)
    plt.xlabel('Time (s)')
    plt.suptitle(f' Subject {subject} - Session {session} \n {suptitle}')
    plt.show()


def pca_comparison(dpca, organized_data_mean , type):
    """
    My intuition with dPCA is that the components should be similar to normal PCA components but with the added
    benefit of labels. In this case the cumulative variance explained by PCA and dPCA should be comparable.
    # This function plots both to allow for this comparison.
    :param dpca:
    :param organized_data_mean:
    :return:
    """
    pca = PCA()
    pca.fit(organized_data_mean.reshape(organized_data_mean.shape[0], -1))
    fig = plt.figure()
    dpca_explained_var_ratios = np.array([dpca.explained_variance_ratio_[key] for key in dpca.explained_variance_ratio_.keys()]).flatten()
    dpca_explained_var_ratios.sort()
    dpca_explained_var_ratios = np.flip(dpca_explained_var_ratios)
    plt.plot(np.cumsum(pca.explained_variance_ratio_[0:len(dpca_explained_var_ratios)]), label='PCA')
    plt.plot(np.cumsum(dpca_explained_var_ratios), label='dPCA')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative variance explained')
    plt.title(f'PCA - {type} vs. dPCA')
    plt.legend()
    plt.show()


# This script serves to perform a dPCA on the spiking data
def sua_prep(subject, session, standardized_data=False, event_lock='Onset'):
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
    if event_lock == 'Onset':
        event_times = onset_times
    elif event_lock == 'Feedback':
        event_times = feedback_times


    # global t-start
    reader = read_file(ph_file_path)
    reader.parse_header()
    start_record = reader.global_t_start
    number_spikes = []
    curr_neuron = 0

    su_data_dir = data_directory / "sorted/sort/final"
    all_su_files = os.listdir(su_data_dir)

    neuron_count = 0
    for file in all_su_files:
        data = loadmat(os.path.join(su_data_dir, file))
        # print(file)
        # print(data['useNegative'])
        # print(data['useNegative'].shape)
        neuron_count += data['useNegative'][0].shape[0]

    file = all_su_files[0]
    max_rt = max(beh_data['response_time'])

    # ahead of grabbing neural data, pre-define some things that are true for every neuron
    binsize = 0.01
    step = 0.01
    tmin_onset = -1
        # tmax = max_rt+1.5
    tmax = 1.5
    trial_time = np.arange(tmin_onset, tmax + step, step)
    n_trials = len(beh_data['correct'])
    n_timepoints = trial_time.shape[0]
    neural_data = np.zeros((n_trials, neuron_count, n_timepoints))
    labels = []

    curr_neur = 0
    for file in all_su_files:

        # step 1. Load in one file, and comb through it for neurons
        microwire_spikes = loadmat(su_data_dir / file)
        neuron_counts = microwire_spikes['useNegative'][0].shape[0]


        for neuron_ind in range(neuron_counts):
            su_cluster_num = microwire_spikes['useNegative'][0][neuron_ind]
            labels.append(str(su_cluster_num))
            # Note that these timestamps are in microseconds, and according to machine clock
            microsec_sec_trans = 10**-6
            su_timestamps = np.array([[microwire_spikes['newTimestampsNegative'][0, i]*microsec_sec_trans-start_record] for i in
                                          range(microwire_spikes['newTimestampsNegative'].shape[1])
                                         if microwire_spikes['assignedNegative'][0, i] == su_cluster_num])

            # trial_wise_feedback_spikes = get_trial_wise_times(su_timestamps, feedback_times, beh_data, tmin=-1., tmax=1.5)
            # Plot response in spikes of this one neuron relative to each onset event
            trial_wise_feedback_spikes = get_trial_wise_times(su_timestamps, event_times, tmin=tmin_onset, tmax=tmax)

            sort_order = sorted(set(beh_data['correct']))
            if len(sort_order) == 3:
                color_dict = dict(zip(sort_order, ['red', 'green', 'blue']))
            else:
                color_dict = dict(zip(sort_order, ['purple', 'orange']))

            spike_trains_sorted, beh_conditions_sorted, change_indices = get_spike_rate_curves(trial_wise_feedback_spikes,
                                                                                               beh_data['correct'],
                                                                                               tmin=tmin_onset,
                                                                                               tmax=tmax, step=step,
                                                                                               binsize=binsize,
                                                                                               mode='single_trial')

            sigma = 0.05
            filtered_signals = gaussian_smooth(spike_trains_sorted, sigma, step)
            neural_data[:, curr_neur, :] = filtered_signals
            curr_neur += 1
            # fig, axs = plt.subplots(nrows=2)
            # plot_neural_spike_trains(axs[0], trial_wise_feedback_spikes, beh_data['correct'], color_dict)
            # axs[0].set_xlim([tmin_onset, tmax])
            # for i in range(filtered_signals.shape[0]):
            #     axs[1].plot(trial_time, filtered_signals[i, :], color=color_dict[beh_conditions_sorted[i]])
            # axs[0].set_title(f"Comparing raster plots and smoothed curves for {su_cluster_num}")
            # axs[1].set_xlabel("Time (s)")
            # axs[1].set_xlim([tmin_onset, tmax])
            # # axs[0].axis('off')
            # plt.show()


    # labels are labels for the single units, consider using the cluster number
    # for ch type, don't use seeg
    eff_fs = int(1/step)
    mne_info = mne.create_info(list(labels), eff_fs, ch_types='seeg')

    # su_data is just what we're plotting above
    epochs_object = mne.EpochsArray(neural_data, mne_info)

    # might take some time to featurize the objects appropriately
    # baselining is a problem
    organized_data_mean, organized_data, feedback_dict = featurize(epochs_object, beh_conditions_sorted,
                                                                   norm=standardized_data)
    return organized_data_mean, organized_data, feedback_dict, trial_time

session = 'sess-2'
subject = 'IR95'
standardized_data = False
event_lock = 'Onset'
organized_data_mean, organized_data, feedback_dict, trial_time = sua_prep(subject, session, standardized_data,
                                                                          event_lock)
plot_signal_avg(organized_data_mean, subject, session, trial_time, labels=feedback_dict,
                extra_string=f'Normalization = {standardized_data} {event_lock}-lock')
regularization_setting = None
# regularization_setting = 'auto'
dpca = dPCA.dPCA(labels='st', regularizer=regularization_setting)
dpca.protect = ['t']
Z = dpca.fit_transform(organized_data_mean, organized_data)

pca_comparison(dpca, organized_data_mean, type='trial average')
new_organized_data = np.swapaxes(organized_data, 0, 1)
new_organized_data_frame = pd.DataFrame(new_organized_data.reshape(new_organized_data.shape[0], -1))
new_organized_data_frame.dropna(inplace=True, axis='columns')
trial_wise_data_for_PCA = new_organized_data_frame.to_numpy()
pca_comparison(dpca, trial_wise_data_for_PCA, type='trial concatenated')
suptitle = f'All single units {event_lock}-locked'
plot_dPCA_components(dpca, Z, trial_time, feedback_dict, subject, session, suptitle,
                     feature_names=['feedback'])

session3 = 'sess-3'
organized_data_mean_3, organized_data_3, feedback_dict_3, trial_time_3 = sua_prep(subject, session3, standardized_data,
                                                                                  event_lock)
plot_signal_avg(organized_data_mean_3, subject, session3, trial_time, labels=feedback_dict,
                extra_string=f'Normalization = {standardized_data}, {event_lock}-locked')

dpca = dPCA.dPCA(labels='st', regularizer=regularization_setting)
dpca.protect = ['t']
Z_2 = dpca.fit_transform(organized_data_mean_3, organized_data_3)

pca_comparison(dpca, organized_data_mean_3, type='trial average')
new_organized_data = np.swapaxes(organized_data_mean_3, 0, 1)
new_organized_data_frame = pd.DataFrame(new_organized_data.reshape(new_organized_data.shape[0], -1))
new_organized_data_frame.dropna(inplace=True, axis='columns')
trial_wise_data_for_PCA = new_organized_data_frame.to_numpy()
pca_comparison(dpca, trial_wise_data_for_PCA, type='trial concatenated')
suptitle = f'All single units {event_lock}-locked'
plot_dPCA_components(dpca, Z_2, trial_time, feedback_dict, subject, session3, suptitle,
                     feature_names=['feedback'])

num_trials_2, num_neurons_2, num_cond, num_timepoints = organized_data.shape
num_trials_3, num_neurons_3, _, _ = organized_data_3.shape
total_neurons = num_neurons_2+num_neurons_3
max_trials_per_cond = np.max([num_trials_2, num_trials_3])
completed_data_set_mean = np.zeros((total_neurons,
                                    num_cond, num_timepoints))
completed_data_set_mean[completed_data_set_mean == 0] = np.nan
completed_data_set_mean[0:organized_data_mean.shape[0], :, :] = organized_data_mean
completed_data_set_mean[organized_data_mean.shape[0]:, :, :] = organized_data_mean_3


completed_data_set = np.zeros((max_trials_per_cond, total_neurons,
                                    num_cond, num_timepoints))
completed_data_set[completed_data_set == 0] = np.nan
completed_data_set[0:num_trials_2, 0:num_neurons_2, :, :] = organized_data
completed_data_set[0:num_trials_3, num_neurons_2:, :, :] = organized_data_3


dpca = dPCA.dPCA(labels='st', regularizer=regularization_setting)
dpca.protect = ['t']
Z = dpca.fit_transform(completed_data_set_mean, completed_data_set)

pca_comparison(dpca, completed_data_set_mean, type='trial average')
new_organized_data = np.swapaxes(completed_data_set_mean, 0, 1)
new_organized_data_frame = pd.DataFrame(new_organized_data.reshape(new_organized_data.shape[0], -1))
new_organized_data_frame.dropna(inplace=True, axis='columns')
trial_wise_data_for_PCA = new_organized_data_frame.to_numpy()
pca_comparison(dpca, trial_wise_data_for_PCA, type='trial concatenated')
suptitle = f'All single units {event_lock}-locked'
session = 'sess-2 and sess-3'
plot_dPCA_components(dpca, Z, trial_time, feedback_dict, subject, session, suptitle,
                     feature_names=['feedback'])
print('huh')