from intracranial_ephys_utils.preprocess import save_small_dataset
import os
from pathlib import Path
import numpy as np
import re
import pandas as pd
import mne
from intracranial_ephys_utils.load_data import read_file, get_file_info
from intracranial_ephys_utils.preprocess import make_trialwise_data, smooth_data
from intracranial_ephys_utils.manual_process import data_clean_viewer
from behavior_analysis import process_wcst_behavior
from scipy.fftpack import fft, ifft, ifftshift, fftshift
import matplotlib.pyplot as plt
from scipy import signal
from fooof import FOOOF


def autocorr(x):
    """
    helper function that returns the autocorrelation of a signal
    :param x:
    :return:
    """
    # nonstationary data so first zscore, then do a little minmax
    yp = (x-np.average(x))/np.std(x)
    min_y = np.min(yp)
    max_y = np.max(yp)
    yp /= (max_y-min_y)
    result = np.correlate(yp, yp, mode='full')
    return result
    # spectrum = np.abs(fft(yp))
    # spectrum *= spectrum
    # x2 = ifft(spectrum)

    # fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    # axs[0].plot(x, y)
    # axs[0].set(xlabel='time, s', ylabel='amplitude, a.u.')
    #
    # axs[1].stem(x_fourier, np.abs(y_fourier))
    # axs[1].set(xlim=(-visual_padding * f, visual_padding * f), xlabel='Frequency, Hz', ylabel='Amplitude, a..u')
    # plt.show()
    # result = np.correlate(x, x, mode='full')

    # basic autocorrelation that depends on ifft
    # xp = ifftshift((x-np.average(x))/np.std(x))
    # n, = xp.shape
    # xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
    # f = fft(xp)
    # p = np.absolute(f)**2
    # power_i = ifft(p)
    # np.real(power_i)[:n//2]/(np.arange(n//2)[::-1]+n//2)

    # third method
    # n = len(x)
    # fx = fft(np.concatenate((xp, np.zeros(n))))  # add zero padding, take FFT
    # x2 = ifft(fx.real**2 + fx.imag**2)  # take absolute square amplitude (power) and invert
    #
    # x2 = np.concatenate((x2[n+2:], x2[:n]))


def autocorr_two(x):
    # nonstationary data so first zscore, then do a little minmax
    yp = (x-np.average(x))/np.std(x)
    min_y = np.min(yp)
    max_y = np.max(yp)
    yp /= (max_y-min_y)
    yp = ifftshift(yp)
    n = yp.shape[0]
    yp = np.r_[yp[:n//2], np.zeros_like(yp), yp[n//2:]]
    f = fft(yp)
    power_yp = np.absolute(f)**2
    autocorrelation_res = ifft(power_yp)
    result = np.real(autocorrelation_res)[:n//2] / (np.arange(n//2)[::-1]+n//2)
    return result
    # y_fourier = np.abs(fftshift(fft(fftshift(yp)))) / len(x)

    # spectrum = np.abs(fft(yp))
    # spectrum *= spectrum
    # x2 = ifft(spectrum)


def auto_correlation(x, lag):
    n = len(x)
    lag_max = int(abs(lag))
    lag_min = -int(abs(lag))
    # Allocate an array for the correlation result for the desired range of lags
    result = np.zeros(int(lag_max - lag_min + 1))

    for lag in range(lag_min, lag_max + 1):
        if lag < 0:
            result[lag - lag_min] = np.dot(x[:n + lag], x[-lag:])
        else:
            result[lag - lag_min] = np.dot(x[lag:], x[:n - lag])

    return result


def lfp_prep(subject, session, task, event_lock='Onset', feature='correct', baseline=(-0.5, 0),
             smooth=False, electrode_selection='all', car=True):
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
    :param electrode_selection: (optional) which electrodes to use
    :param car: (optional) Whether to use common average reference on, default True
    :return: epochs_object (Epochs) MNE epochs object
    :return: trial_time (array) Associated timepoints with data in epochs object
    :return: electrode_names (array) Names for each electrode in epochs object
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
        tmin = -0.5
        baseline = baseline
        tmax_actual = tmax
        tmin_actual = tmin
    elif event_lock == 'Feedback':
        event_times = feedback_times
        tmin = -0.5
        tmax = 2.5
        baseline = baseline
        tmax_actual = tmax - 1.
        # tmax = 1.5
        tmax_actual = tmax
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
    dropped_electrodes_noisy = sbj_metadata[subject][session]['dropped_electrodes_noisy']
    dropped_electrodes_reference = sbj_metadata[subject][session]['dropped_electrodes_reference']
    dropped_electrodes_micro_PED = sbj_metadata[subject][session]['dropped_electrodes_microperiodic_epileptic_discharge']
    oob_electrodes = sbj_metadata[subject][session]['dropped_electrodes_oob']
    wm_electrodes = sbj_metadata[subject][session]['dropped_electrodes_wm']
    macro_noisy = sbj_metadata[subject][session]['dropped_macros_noisy']
    no_reference_electrodes = sbj_metadata[subject][session]['dropped_macros_no_reference']
    dropped_electrodes = dropped_electrodes_reference + dropped_electrodes_noisy + dropped_electrodes_micro_PED
    dropped_macrocontacts = oob_electrodes + no_reference_electrodes + macro_noisy + wm_electrodes
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
    print('dropping reference microwire electrode and noisy bundles')
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
            if car:
                # common average reference for microwires
                micro_contacts = [ind for ind, val in enumerate(electrode_names) if val.startswith(probe)]
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

    # get index for which electrodes should be dropped, but drop these after referencing
    electrodes_ind = [ind for ind in range(electrode_names.size) if electrode_names[ind] not in
                      dropped_macrocontacts]
    electrode_names = electrode_names[electrodes_ind]
    dataset = dataset[electrodes_ind, :]
    print('dropping white matter and out of brain electrodes')
    print(dataset.shape)
    # following rereferencing we'll band pass filter and notch filtering to remove HF noise and power line noise for
    # both macrocontacts and microwires
    h_freq = 200
    l_freq = 1
    neural_ind = [i for i, element in enumerate(electrode_names) if not any(element.startswith(skip) for skip in
                                                                            skippables)]
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

    # select electrodes
    if electrode_selection == 'microwire':
        electrode_ind = [i for i, electrode_name in enumerate(electrode_names) if electrode_name.startswith('m')
                         and not electrode_name.startswith('mic')]
        electrode_names = [electrode_name for i, electrode_name in enumerate(electrode_names) if
                           electrode_name.startswith('m')
                           and not electrode_name.startswith('mic')]
    elif electrode_selection == 'macrocontact':
        electrode_ind = [i for i, electrode_name in enumerate(electrode_names) if
                         not (electrode_name.startswith('m') or
                              np.any([skippables[i].startswith(electrode_name[:3]) for i in range(len(skippables))]))]
        electrode_names = [electrode_name for i, electrode_name in enumerate(electrode_names) if
                           not (electrode_name.startswith('m') or
                                np.any([skippables[i].startswith(electrode_name[:3]) for i in range(len(skippables))]))]
    else:
        electrode_ind = [i for i, electrode_name in enumerate(electrode_names) if not
                         np.any([skippables[i].startswith(electrode_name[:3]) for i in range(len(skippables))])]
        electrode_names = [electrode_name for i, electrode_name in enumerate(electrode_names) if not
                           np.any([skippables[i].startswith(electrode_name[:3]) for i in range(len(skippables))])]
    print(electrode_names)
    lfp_dataset = dataset[electrode_ind, :]

    assert len(set(fs)) == 1
    sampling_rate = fs[0]

    # # look at data post filtering to see if uPEDs are present
    # # data_directory = Path(f'{os.pardir}/data/{subject}/{session}')
    # # data_clean_viewer(subject, session, task, data_directory, electrode_names, lfp_dataset, int(fs[0]))
    # no_ped_ind = 30
    # ped_ind = 33
    # ped_electrode_lfp = lfp_dataset[ped_ind, :]
    # no_ped_electrode_lfp = lfp_dataset[no_ped_ind, :]
    # auto_correlo_time = 2
    # lag = auto_correlo_time*sampling_rate
    # correlation_ped = auto_correlation(ped_electrode_lfp, lag)
    # correlation_no_ped = auto_correlation(no_ped_electrode_lfp, lag)
    # fig, ax = plt.subplots()
    # d_time = np.arange(-auto_correlo_time, auto_correlo_time+round(1/sampling_rate, 4), round(1/sampling_rate, 4))
    # plt.plot(d_time, correlation_ped, color='green', label='ped')
    # plt.plot(d_time,correlation_no_ped, color='blue', label='no ped')
    # # tmax = 5  # seconds
    # # d_time = np.arange(0, tmax, round(1 / sampling_rate, 4))
    # # plt.plot( ped_[:int(tmax * sampling_rate)], color='blue', label='fast')
    # plt.legend()
    # plt.title(f'Autocorrelolgram of {electrode_names[ped_ind] and electrode_names[no_ped_ind]}')
    # # plt.xlim([0, int(10*sampling_rate)])
    # # plt.ylim([-10,10])
    # plt.show()
    #
    # # plot power spectrum
    # first_ind = 30
    # fft_values_noped = np.fft.fft(lfp_dataset[first_ind, :])
    # fft_freqs_noped = np.fft.fftfreq(len(lfp_dataset[first_ind, :]), round(1 / sampling_rate, 4))
    #
    # # power spectrum (magnitude squared of FFT)
    # power_spectrum_noped = np.abs(fft_values_noped)**2
    #
    # # Only plot the positive half of the spectrum (symmetry in FFT)
    # half_n = len(dataset[first_ind, :]) // 2
    # plt.plot(fft_freqs_noped[:half_n], power_spectrum_noped[:half_n])
    # plt.title(f'Power Spectrum {electrode_names[first_ind]}')
    # plt.xlabel('Frequency (Hz)')
    # plt.xlim([0, 20])
    # plt.ylabel('Power')
    # plt.grid(True)
    # plt.show()
    #
    # second_ind = 33
    # fft_values_PED = np.fft.fft(lfp_dataset[second_ind, :])
    # fft_freqs_PED = np.fft.fftfreq(len(lfp_dataset[second_ind, :]), round(1 / sampling_rate, 4))
    #
    # # power spectrum (magnitude squared of FFT)
    # power_spectrum_PED = np.abs(fft_values_PED) ** 2
    #
    # # Only plot the positive half of the spectrum (symmetry in FFT)
    # half_n = len(lfp_dataset[second_ind, :]) // 2
    # plt.plot(fft_freqs_PED[:half_n], power_spectrum_PED[:half_n])
    # plt.title(f'Power Spectrum {electrode_names[second_ind]}')
    # plt.xlabel('Frequency (Hz)')
    # plt.xlim([0, 20])
    # plt.ylabel('Power')
    # plt.grid(True)
    # plt.show()
    #
    # lowfreq = 1
    # high_freq = 2
    # butterworth_bandpass = signal.butter(4, (lowfreq, high_freq), 'bp', fs=sampling_rate, output='sos')
    # low_ped_freq = 8
    # high_ped_freq = 20
    # ped_bandpass = signal.butter(4, (low_ped_freq, high_ped_freq), 'bp', fs=sampling_rate,
    #                                      output='sos')
    # lfp_noPED = lfp_dataset[first_ind, :]
    # lfp_wPED = lfp_dataset[second_ind, :]
    # heartrate_signal_noPED = signal.sosfiltfilt(butterworth_bandpass, lfp_noPED)
    # heartrate_signal_wPED = signal.sosfiltfilt(butterworth_bandpass, lfp_wPED)
    # ped_signal_noPED = signal.sosfiltfilt(ped_bandpass, lfp_noPED)
    # ped_signal_PED = signal.sosfiltfilt(ped_bandpass, lfp_wPED)
    # power_noPED = np.sum(heartrate_signal_noPED**2)
    # power_wPED = np.sum(heartrate_signal_wPED**2)
    # print(power_noPED)
    # print(power_wPED)
    # tmax = 80
    # ind_max = int(tmax*sampling_rate)
    # fig, ax = plt.subplots(3, sharex=True)
    # time_x = np.arange(0, tmax, round(1/sampling_rate, 4))
    # ax[0].plot(time_x, lfp_noPED[0:ind_max], label='No PED', color='red', alpha=0.5)
    # ax[0].plot(time_x, lfp_wPED[0:ind_max], label='With PED', color='blue', alpha=0.5)
    # ax[1].plot(time_x, heartrate_signal_noPED[0:ind_max], label='No PED', color='red')
    # ax[1].plot(time_x, heartrate_signal_wPED[0:ind_max], label='With PED', color='blue')
    # ax[1].set_title(f'Bandpass from {lowfreq} to {high_freq} ')
    # ax[2].plot(time_x, ped_signal_noPED[0:ind_max], label='No PED', color='red', alpha=0.5)
    # ax[2].plot(time_x, ped_signal_PED[0:ind_max], label='With PED', color='blue', alpha=0.5)
    # ax[2].set_title(f'Bandpass from {low_ped_freq} to {high_ped_freq} ')
    # ax[2].set_xlabel('Time (seconds)')
    # plt.legend()
    # plt.suptitle(f'Comparison between signals {electrode_names[first_ind]} and {electrode_names[second_ind]}')
    # plt.tight_layout()
    # plt.show()
    #
    # # fit fooof to get peaks, and see if peaks are in the range seen by our histogram
    # # Import the FOOOF object
    #
    # # Initialize FOOOF object
    # fm = FOOOF()
    #
    # # Define frequency range across which to model the spectrum
    # freq_range = [0.5, 70]
    #
    # # Model the power spectrum with FOOOF, and print out a report
    # fm.report(fft_freqs_noped[:half_n],  power_spectrum_noped[:half_n], freq_range)
    #
    #
    # fm.report(fft_freqs_PED[:half_n],  power_spectrum_PED[:half_n], freq_range)


    # # taken from 2024 Rutishauser paper on human hippocampal neurons, Nature
    # mean_lfp_dataset = np.mean(lfp_dataset, axis=0)
    # std_lfp_dataset = np.std(lfp_dataset, axis=0)
    # zscored_lfp = (lfp_dataset - mean_lfp_dataset) / std_lfp_dataset
    # zscored_lfp[zscored_lfp >= 6] = 6.
    # zscored_lfp[zscored_lfp <= -6] = -6.
    # mean_zscored_lfp_dataset = np.mean(zscored_lfp, axis=0)
    # std_zscored_lfp_dataset = np.std(zscored_lfp, axis=0)
    # rezscored_lfp = (zscored_lfp - mean_zscored_lfp_dataset) / std_zscored_lfp_dataset
    # artifact_indices = [i for i in range(zscored_lfp.shape[1]) if np.any(abs(rezscored_lfp[:, i]) > 4)]
    #
    # dilation_factor = 0.050
    # window_size = int(dilation_factor * fs[0])  # 50 msec
    # # next loop through this data and find where runs end
    #
    # start_events = []
    # stop_events = []
    # i = 0
    # while i < len(artifact_indices):
    #     # Start a new event
    #     start_events.append(artifact_indices[i])
    #
    #     # Initialize stop value to the current artifact index
    #     current_stop_value = artifact_indices[i]
    #
    #     # Check for any subsequent timestamps within the 50 ms window
    #     while (i + 1 < len(artifact_indices) and
    #            artifact_indices[i + 1] <= current_stop_value + window_size):
    #         i += 1
    #         current_stop_value = artifact_indices[i]  # Expand the stop value
    #
    #     # Once no more timestamps are within 50 ms, mark the stop of the event
    #     stop_events.append(current_stop_value)
    #
    #     # Move to the next potential event
    #     i += 1
    #
    # artifact_durations = np.round([stop_events[i] - start_events[i] for i in range(len(stop_events))] / sampling_rate,
    #                               4)
    # # remove anything that's too short to matter, say 50 msec
    # start_events = np.array(start_events)[artifact_durations > dilation_factor]
    # artifact_durations = artifact_durations[artifact_durations > dilation_factor]
    # start_artifacts = np.round(start_events / sampling_rate, 4)
    # artifacts_descriptions = len(artifact_durations) * ['bad artifacts']
    # our .csv file containing timestamps does have it in terms of samples but expecting sampling rate of photodiode
    # which has now been perturbed
    # instead we'll take the time in seconds, subtract the time at the start of the task to reference that to 0, and
    # convert to samples
    event_times_converted = ((event_times.copy() - timestamps[0]) * sampling_rate).astype(int)

    data_directory = Path(f'{os.pardir}/data/{subject}/{session}')
    # check for annotations and apply them
    file_path = data_directory / f'{subject}_{session}_{task}_post_timestamping_events.csv'
    # annotations = None
    if os.path.isfile(file_path):
        annotations_df = pd.read_csv(file_path)
        onsets = list(annotations_df['time']) #+ list(start_artifacts)
        durations = list(annotations_df['duration']) #+ list(artifact_durations)
        description = list(annotations_df['label']) #+ list(artifacts_descriptions)
        # durations = [x for _, x in sorted(zip(onsets, durations))]
        # description = [x for _, x in sorted(zip(onsets, description))]
        # onsets = sorted(onsets)
        annotations = mne.Annotations(onsets, durations, description)
        better_annotations_df = pd.DataFrame(onsets, columns=['onsets'])
        better_annotations_df['durations'] = durations
        better_annotations_df['description'] = description
    else:
        annotations = None
    epochs_object = make_trialwise_data(event_times_converted, electrode_names, sampling_rate, lfp_dataset,
                                        tmin=tmin,
                                        tmax=tmax,
                                        baseline=baseline, annotations=annotations)

    # generate timestamps for data
    # trial_time_full = np.arange(tmin+step, tmax, step)
    # do a zscore style analysis here to remove some data
    print(epochs_object.get_data())
    print(epochs_object.drop_log)
    epochs_object.drop_bad()
    print(epochs_object.drop_log)
    # drop the associated label with a trial if trial was dropped
    retained_ind = [ind for ind in range(len(feature_values)) if len(epochs_object.drop_log[ind]) == 0]
    feature_values = feature_values[retained_ind]

    # Smooth and downsample data if analyzing broadband, otherwise create the object to allow further processing
    # careful that smoothing removes annotation as part of the epochs_object
    if smooth:
        binsize = 0.1
        step = 0.05
        epochs_dataset = epochs_object.get_data()
        # trim the end of the epoch (this should be change if wanting to trim something else)
        trial_len_samples = (tmax_actual-tmin) * sampling_rate + 1
        trial_time = np.round(np.arange(tmin+step, tmax_actual, step), 2)
        epochs_dataset = epochs_dataset[:, :, :int(trial_len_samples)]
        smoothed_data, fs = smooth_data(epochs_dataset, sampling_rate, binsize, step)
        mne_info = mne.create_info(electrode_names, fs, ch_types='seeg')
        epochs_object = mne.EpochsArray(smoothed_data, mne_info)
    else:
        step_fs = 1 / sampling_rate
        trial_time = np.arange(tmin, tmax+step_fs, step_fs)
    return epochs_object, trial_time, electrode_names, feature_values


def main():
    test_subject = 'IR95'
    test_session = 'sess-3'
    test_task = 'wcst'
    events_file_name = f'{test_subject}_{test_session}_{test_task}_events.csv'
    events_path = Path(f'{os.pardir}/data/{test_subject}/{test_session}/{events_file_name}')
    # save_small_dataset(test_subject, test_session, task_name=test_task, events_file=events_path)


if __name__ == "__main__":
    main()
