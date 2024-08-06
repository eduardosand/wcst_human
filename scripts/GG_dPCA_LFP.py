import numpy as np
import os
from pathlib import Path
from intracranial_ephys_utils.load_data import read_file, get_file_info
from intracranial_ephys_utils.preprocess import make_trialwise_data, smooth_data
from behavior_analysis import process_wcst_behavior
from G_dPCA import featurize, plot_signal_avg, dpca_plot_analysis, trial_wise_processing
import re
import pandas as pd
import mne


def lfp_prep(subject, session, task, event_lock='Onset', feature='correct', baseline=(-0.5, 0),
             smooth=False):
    """
    Prepare biopotential data for further analyses. Expects preprocessed bandpassed signal at sampling rate of 1000.
    From here, we'd like to find the onsets of the trials and smooth over them as in Hoy et. al.
    :param subject: (string) ID of the subject.
    :param session: (string) ID of the session.
    :param task: (string) what task, used for loading data
    :param event_lock: (optional) Can lock to onsets or offsets
    :param feature: (optional) automatically, correct. In theory could use anything from behavior data
    :param baseline: (optional) helpful for toying with baselines
    :param smooth: (optional) whether to smooth data or not
    :return: epochs_object (Epochs) MNE epochs object
    :return: trial_time (array) Associated timepoints with data in epochs object
    :return: microwire_names (array) Names for each electrode in epochs object
    :return: feature_values (array) Associated condition for each
    """

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
        tmax = 1.5
        tmin = -1
        baseline = baseline
        tmax_actual = tmax
        tmin_actual = tmin
    elif event_lock == 'Feedback':
        event_times = feedback_times
        tmin = -0.5
        tmax = 2.5
        baseline = baseline
        tmax_actual = tmax - 1.
        tmin_actual = tmin

    # global t-start
    reader = read_file(ph_file_path)
    reader.parse_header()
    # start_record = reader.global_t_start

    bp = 1000
    processed_data_directory = data_directory / 'preprocessed'
    dataset_path = processed_data_directory / f'{subject}_{session}_{task}_lowpass_{bp}.npz'
    full_dataset = np.load(dataset_path)
    print('dataset loaded')
    dataset = full_dataset['dataset'][:-1, :]
    timestamps = full_dataset['dataset'][-1, :]  # this should be in seconds from start of recording(start of file)
    fs = full_dataset['eff_fs'][:-1]
    electrode_names = full_dataset['electrode_names'][:-1]

    # dropped_electrodes =
    import json
    file_path = Path(f'{os.pardir}/scripts/subject_deets.json')
    with open(file_path) as json_data:
        sbj_metadata = json.load(json_data)
    dropped_electrodes = sbj_metadata[subject][session]['dropped_electrodes']
    # prior to dropping
    print(dataset.shape)


    # issue with electrode_names is sometimes there's an extra ending to denote the recording number, it should be
    # present on all channels so we get rid of it by using the first name
    stems = electrode_names[0].split("_")

    # Now we have the dataset, next step is to rereference
    electrode_names_str = " ".join(electrode_names)

    electrode_names_str = re.sub(f"_{stems[-1]}", "", electrode_names_str)
    electrode_names = np.array(electrode_names_str.split(" "))
    # get index for which electrodes should be dropped
    electrodes_ind = [ind for ind in range(electrode_names.size) if electrode_names[ind] not in dropped_electrodes]
    electrode_names = electrode_names[electrodes_ind]
    dataset = dataset[electrodes_ind, :]
    print('dropping reference electrodes')
    print(dataset.shape)

    # Now we have the dataset, next step is to rereference
    #
    electrode_names_str = " ".join(electrode_names)

    electrode_names_str = re.sub(f"_{stems[-1]}", "", electrode_names_str)
    electrode_names_fixed = re.sub("\d+", "", electrode_names_str)
    skippables = ['spk', 'mic', 'eye', 'photo']
    for probe in set(electrode_names_fixed.split(" ")):
        if probe in skippables:
            continue
        # Assume all these electrodes are numbered from 1 - end, with 1 being the tip(deepest part of the brain).
        num_contacts = electrode_names_fixed.split(" ").count(probe)
        if probe.startswith('m'):
            # common average reference for microwires
            micro_contacts = [ind for ind, val in enumerate(electrode_names) if val.startswith(probe)]
            # print(micro_contacts)
            common_avg_ref = np.average(dataset[micro_contacts, :], axis=0)
            dataset[micro_contacts, :] -= common_avg_ref
            continue
        elif num_contacts > 1:
            # bipolar reference for everything else
            for contact_num in range(num_contacts - 1):
                curr_electrode = f"{probe}{contact_num + 1}"
                electrode_ind = np.where(electrode_names == curr_electrode)[0][0]
                next_electrode_ind = np.where(electrode_names == f"{probe}{contact_num + 2}")[0][0]
                dataset[electrode_ind, :] -= dataset[next_electrode_ind, :]
            continue

    # following rereferencing we'll band pass filter and notch filtering to remove HF noise and power line noise for
    # both macrocontacts and microwires
    h_freq = 200
    l_freq = 1
    neural_ind = [i for i, element in enumerate(electrode_names) if not any(element.startswith(skip) for skip in skippables)]
    filtered_neural_data = mne.filter.filter_data(dataset[neural_ind], fs[0], l_freq, h_freq)

    # notch filter
    filtered_neural_data = mne.filter.notch_filter(filtered_neural_data, fs[0], np.arange(60, 241, 60))

    # cast this data back to dataset array
    dataset[neural_ind] = filtered_neural_data


    if feature == 'correct':
        beh_data["correct"] = np.where(beh_data["correct"] == 0, 'incorrect', 'correct')
        feature_values = beh_data['correct']
    elif feature == 'rule dimension':
        feature_values = beh_data['rule dimension']
    else:
        feature_values = beh_data['chosen']
    sort_order = sorted(set(feature_values))
    if len(sort_order) == 3:
        color_dict = dict(zip(sort_order, ['red', 'green', 'blue']))
    else:
        color_dict = dict(zip(sort_order, ['purple', 'orange']))

    # This part of the code should do the smoothing
    microwire_ind = [i for i, electrode_name in enumerate(electrode_names) if electrode_name.startswith('m')
                     and not electrode_name.startswith('mic')]
    microwire_names = [electrode_name for i, electrode_name in enumerate(electrode_names) if
                       electrode_name.startswith('m')
                       and not electrode_name.startswith('mic')]

    microwire_dataset = dataset[microwire_ind, :]

    # our .csv file containing timestamps does have it in terms of samples but expecting sampling rate of photodiode
    # which has now been perturbed
    # instead we'll take the time in seconds, subtract the time at the start of the task to reference that to 0, and
    # convert to samples
    assert len(set(fs)) == 1
    sampling_rate = fs[0]
    event_times_converted = ((event_times.copy() - timestamps[0]) * sampling_rate).astype(int)
    epochs_object = make_trialwise_data(event_times_converted, microwire_names, sampling_rate, microwire_dataset,
                                        tmin=tmin,
                                        tmax=tmax,
                                        baseline=baseline)

    # generate timestamps for data
    # trial_time_full = np.arange(tmin+step, tmax, step)

    # compute time frequency representations if HFA is of interest
    # if foi == 'HFA':
    #     step_fs = 1 / sampling_rate
    #     trial_time_full = np.arange(tmin, tmax+step_fs, step_fs)
    #     tfr_power = multitaper(subject, session, task, epochs_object, foi='HFA')
    #     HFA_power_normalized = log_normalize(tfr_power, trial_time_full, baseline=baseline)
    #     epochs_object = mne.EpochsArray(HFA_power_normalized, mne_info)

    print(epochs_object.drop_log)

    # Smooth and downsample data if analyzing broadband, otherwise create the object to allow further processing
    if smooth:
        binsize = 0.1
        step = 0.05
        epochs_dataset = epochs_object.get_data()
        # trim the end of the epoch (this should be change if wanting to trim something else)
        trial_len_samples = (tmax_actual-tmin) * sampling_rate + 1
        trial_time = np.arange(tmin+step, tmax_actual, step)
        epochs_dataset = epochs_dataset[:, :, :int(trial_len_samples)]
        smoothed_data, fs = smooth_data(epochs_dataset, sampling_rate, binsize, step)
        mne_info = mne.create_info(microwire_names, fs, ch_types='seeg')
        epochs_object = mne.EpochsArray(smoothed_data, mne_info)
    else:
        step_fs = 1 / sampling_rate
        trial_time = np.arange(tmin, tmax+step_fs, step_fs)

    return epochs_object, trial_time, microwire_names, feature_values


def organize_data(epochs_object, feature_values, standardized_data=False,
                  method='else'):
    """
    Idea for this function is take the output from lfp_prep and set up data for different analyses, namely, PCA, dPCA,
    HFA etc
    :param epochs_object: (Epoch): MNE Epoch object, data in here should be of
    shape (n_trials X n_electrodes X n_timepoints)
    :param microwire_names: (list): list of microwire names, should match in size to n_electrodes
    :param feature_values:
    :param standardized_data: (optional) Whether to normalized the data or not
    :param method: (optional) - whether to organize the data for raw PCA, dPCA or other
    :return: organized_data_mean: (ndarray) - shape (n_electrodes, n_cond, n_timepoints)
    :return: organized_data: (ndarray) - might be different shape depending on method
    :return: feedback_dict: (dict)
    """
    organized_data_mean, organized_data, feedback_dict = featurize(epochs_object, feature_values,
                                                                   norm=standardized_data)
    if method == 'PCA':
        zscored_data = trial_wise_processing(epochs_object, norm=standardized_data)
    elif method == 'dPCA':
        zscored_data = organized_data
    else:
        zscored_data = organized_data
    return organized_data_mean, zscored_data, feedback_dict


def multitaper(sbj, session, task, epochs, foi='HFA'):
    """
    This function computes a time frequency decomposition of electrophysiological, in such a way
    to obtain high frequency broadband power(HF BB). We take a series of band ranges from 70-160.

    :param sbj: (string) : Subject Identifier
    :param session: (string): Session Identifier
    :param task: (string): Task Identifier
    :param epochs: (Epoch) : MNE Epoch object
    :return: tfr_power (Power) : Time Frequency Decomposition of epoched data, without averaging.
                                Data in this class will be an array of following dimensions
                                (n_epochs, n_electrodes, n_freq, n_timepoints)
    """
    # decim_parameter = 2 # Factor for decimation
    # other variables used previously not using now use_fft=True and verbose=None
    if foi == 'HFA':
        band_range = np.arange(70, 160, 10)
        n_cycles_inst = band_range/2
        time_bandwidth_inst = 7

        path_directory = Path(f'{os.pardir}/data/{sbj}/{session}/preprocessed/')


        file_path = path_directory / f"{sbj}_{session}_{task}_multitaper_HFA_decomposition-tfr.h5"

        tfr_power = epochs.compute_tfr(method='multitaper', freqs=band_range, n_cycles=n_cycles_inst,
                                       time_bandwidth=time_bandwidth_inst, return_itc=False, average=False, n_jobs=1,
                                       use_fft=True)
        tfr_power.save(file_path, overwrite=True)
    else:
        # freqs = np.arange(5.0, 100.0, 3.0)
        vmin, vmax = -3.0, 3.0  # Define our color limits.
        band_range = np.arange(4., 160., 4)
        n_cycles_inst = band_range/2
        time_bandwidth_inst = 7

        path_directory = Path(f'{os.pardir}/data/{sbj}/{session}/preprocessed/')


        file_path = path_directory / f"{sbj}_{session}_{task}_multitaper_HFA_decomposition-tfr.h5"

        tfr_power = epochs.compute_tfr(method='multitaper', freqs=band_range, n_cycles=n_cycles_inst,
                                       time_bandwidth=time_bandwidth_inst, return_itc=False, average=False, n_jobs=1,
                                       use_fft=True)
        tfr_power.save(file_path, overwrite=True)

    # tfr_power = tfr_multitaper(epochs, band_range, n_cycles_inst, time_bandwidth=time_bandwidth_inst,
    # use_fft=True, return_itc=False, decim=decim_parameter, average=False,
    # 											  verbose=None, n_jobs=1)

    return tfr_power


def morlet(sbj, session, task, epochs):
    """
    Perform a time frequency decomposition using Morlet wavelets. Use fewer cycles on higher frequencies
    This serves to get a quick glance at the bands that may be involved in task.
    :param sbj: (string) : Subject Identifier
    :param session: (string): Session Identifier
    :param task: (string): Task Identifier
    :param epochs: (Epoch) : MNE Epoch object
    :return: tfr_power (Power) : Time Frequency Decomposition of epoched data, without averaging.
                                Data in this class will be an array of following dimensions
                                (n_epochs, n_electrodes, n_freq, n_timepoints)
    """
    # freqs = np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 21, 24, 27, 30, 35,
    #                   40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180])
    # bwidth = np.array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 5, 5,
    #                    5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20])
    freqs = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 21, 24, 27, 30, 35,
                      40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 220, 240])
    bwidth = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 5, 5,
                       5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20])

    freqs = np.array(freqs)
    bwidth = np.array(bwidth)

    time_bandwidth = 2
    n_cycles = freqs * time_bandwidth / bwidth
    tfr_power = epochs.compute_tfr(
        method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True
    )
    path_directory = Path(f'{os.pardir}/data/{sbj}/{session}/preprocessed/')

    file_path = path_directory / f"{sbj}_{session}_{task}_morlet_decomposition-tfr.h5"
    tfr_power.save(file_path, overwrite=True)
    return tfr_power


def log_normalize(tfr_power, trial_time, baseline=(2, 2.5)):
    """
    This function takes time frequency decomposition of epoched data and first log normalizes according to
        a baseline of the first second of data. Then the data is averaged across frequency bands to obtain an
        estimate of high frequency broadband (HF BB) power. NOTE: This script was written with compute limitations
        in mind (memory). As such, it might not be as 'pythonic' as it could be.

    :param tfr_power: (Power) : Time Frequency Decomposition of epoched data, without averaging.
                                Data in this class will be an array of following dimensions
                                (n_epochs, n_electrodes, n_freq, n_timepoints)
    :param trial_time: (array) : timestamps for each timepoint in tfr in seconds (event_locked)
    :param baseline: (tuple) : tells the period to use as baseline in seconds with reference to the event, used with
    trial time
    :return: HFA_power_normalized (array): Array containing HF BB power, trial and electrode wise.
                                           Array is of the following dimensions
                                           (n_epochs, n_electrodes, n_timepoints)
    """

    (n_trials, n_electrodes, n_freq, n_timepoints) = tfr_power.data.shape
    HFA_power_normalized = np.zeros((n_trials, n_electrodes, n_timepoints))
    baseline_start = int(np.argmax(trial_time >= baseline[0]))
    baseline_end = int(np.argmax(trial_time >= baseline[1]))
    for i in range(n_freq):
        mean_baseline = np.mean(np.log(tfr_power.data[:, :, i, baseline_start:baseline_end]), axis=(0, 2),
                                keepdims=True)
        std = np.std(np.log(tfr_power.data[:, :, i, baseline_start:baseline_end]), axis=(0, 2), keepdims=True)
        power = (np.log(tfr_power.data[:, :, i, :])-mean_baseline) / std
        HFA_power_normalized += power
    HFA_power_normalized /= n_freq
    return HFA_power_normalized


def main():
    # plan
    # this code will load in the dataset we downloaded, and attempt to fit a simple dPCA model using just the feedback
    # signal
    test_subject = 'IR95'
    test_session = 'sess-3'
    task = 'wcst'
    bp = 1000
    event_lock = 'Feedback'
    feature = 'correct'
    standardized_data = False
    # regularization_setting = 'auto'
    regularization_setting = None
    epochs_dataset, trial_time, microwire_names, feature_values = lfp_prep(test_subject, test_session, task,
                                                                           event_lock=event_lock, feature=feature)
    organized_data_mean, organized_data, feedback_dict = organize_data(epochs_dataset, feature_values,
                                                                       standardized_data=standardized_data,
                                                                       method='dPCA')

    feature_dict = feedback_dict

    # suptitle=f'All microwires, bandpassed at {bp}, {lock}-locked'
    plot_signal_avg(organized_data_mean, test_subject, test_session, trial_time, labels=feedback_dict,
                        extra_string=f'Normalization = {standardized_data} {event_lock}-lock',
                    signal_names=microwire_names)
    dpca_1, Z_1 = dpca_plot_analysis(organized_data_mean, organized_data, trial_time, feature_dict, test_subject,
                                     test_session, event_lock, standardized_data,
                                     regularization_setting=regularization_setting,
                                     feature_names=[feature], data_modality='Microwire broadband')

    standardized_data = True
    organized_data_mean, organized_data, feedback_dict = organize_data(epochs_dataset, feature_values,
                                                                       standardized_data=standardized_data,
                                                                       method='dPCA')
    feature_dict = feedback_dict

    # suptitle=f'All microwires, bandpassed at {bp}, {lock}-locked'
    plot_signal_avg(organized_data_mean, test_subject, test_session, trial_time, labels=feedback_dict,
                        extra_string=f'Normalization = {standardized_data} {event_lock}-lock')
    dpca_2, Z_2 = dpca_plot_analysis(organized_data_mean, organized_data, trial_time, feature_dict, test_subject,
                                     test_session, event_lock, standardized_data,
                                     regularization_setting=regularization_setting,
                                     feature_names=[feature], data_modality='Microwire broadband')


if __name__ == "__main__":
    main()
