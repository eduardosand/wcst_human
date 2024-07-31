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


def lfp_prep(subject, session, task, standardized_data=False, event_lock='Onset', feature='correct',
             for_dpca=False):
    """
    Prepare biopotential data for further analyses. Expects preprocessed bandpassed signal at sampling rate of 1000.
    From here, we'd like to find the onsets of the trials and smooth over them as in Hoy et. al.
    :param subject: (string) ID of the subject.
    :param session: (string) ID of the session.
    :param task: (task)
    :param standardized_data:
    :param event_lock: Can lock to onsets or offsets
    :param feature:
    :param frequency_cutoff:
    :param diagnostic:
    :param for_dpca: (optional)
    :return:
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
        baseline = (-1, 0)
    elif event_lock == 'Feedback':
        event_times = feedback_times
        tmin = -0.5
        tmax = 2.5
        baseline = (2, 2.5)

    # global t-start
    reader = read_file(ph_file_path)
    reader.parse_header()
    start_record = reader.global_t_start

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

    # electrode_names_str = re.sub("m.... ", "", electrode_names_str)
    electrode_names_fixed = re.sub("\d+", "", electrode_names_str)
    skippables = ['spk', 'mic', 'eye']
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

    max_rt = max(beh_data['response_time'])

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
    # stimulus_onset = ph_onset_trials[~np.isnan(padded_rts)] / ph_sample_rate * fs
    # feedback_onset = ph_offset_button_press[~np.isnan(padded_rts)] / ph_sample_rate * fs
    print('Converted event times')
    epochs_object = make_trialwise_data(event_times_converted, microwire_names, sampling_rate, microwire_dataset,
                                        tmin=tmin,
                                        tmax=tmax,
                                        baseline=baseline)
    epochs_dataset = epochs_object.get_data()
    # photodiode_ind = np.where(electrode_names == 'photo1')[0][0]
    print(epochs_object.drop_log)
    n_epochs, n_electrodes, n_timepoints = epochs_dataset.shape

    binsize = 0.1
    step = 0.05

    trial_time = np.arange(tmin+step, tmax, step)
    n_trials = len(beh_data['correct'])
    smoothed_data, fs = smooth_data(epochs_dataset, sampling_rate, binsize, step)

    mne_info = mne.create_info(microwire_names, fs, ch_types='seeg')
    epochs_object = mne.EpochsArray(smoothed_data, mne_info)
    print(epochs_object)
    epochs_dataset = epochs_object.get_data(copy=True)
    n_epochs, n_electrodes, n_timepoints = epochs_dataset.shape
    # might take some time to featurize the objects appropriately
    # baselining is a problem
    organized_data_mean, organized_data, feedback_dict = featurize(epochs_object, feature_values,
                                                                   norm=standardized_data)
    if not for_dpca:
        zscored_data = trial_wise_processing(epochs_object, norm=standardized_data)
        print(zscored_data.shape)
    else:
        zscored_data = organized_data
    return organized_data_mean, zscored_data, feedback_dict, trial_time, microwire_names, feature_values


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
    organized_data_mean, organized_data, feedback_dict, trial_time, microwire_names = lfp_prep(test_subject, test_session,
                                                                                               task, event_lock=event_lock,
                                                                                               feature=feature,
                                                                                               standardized_data=standardized_data)
    feature_dict = feedback_dict

    # suptitle=f'All microwires, bandpassed at {bp}, {lock}-locked'
    plot_signal_avg(organized_data_mean, test_subject, test_session, trial_time, labels=feedback_dict,
                        extra_string=f'Normalization = {standardized_data} {event_lock}-lock', signal_names=microwire_names)
    dpca_1, Z_1 = dpca_plot_analysis(organized_data_mean, organized_data, trial_time, feature_dict, test_subject,
                                     test_session, event_lock, standardized_data,
                                     regularization_setting=regularization_setting,
                                     feature_names=[feature], data_modality='Microwire broadband')

    standardized_data = True
    organized_data_mean, organized_data, feedback_dict, trial_time, microwire_names = lfp_prep(test_subject, test_session, task,
                                                                              event_lock=event_lock, feature=feature,
                                                                              standardized_data=standardized_data)
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
