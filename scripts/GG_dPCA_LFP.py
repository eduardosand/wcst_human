import numpy as np
import os
from pathlib import Path
from BB_processlfp import plot_signal_avg
from G_dPCA import dpca_plot_analysis

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
