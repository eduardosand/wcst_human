# This code exists to perform a PCA analysis of the broadband LFP data
from GG_dPCA_LFP import lfp_prep, organize_data, multitaper, log_normalize, morlet
from G_dPCA import plot_signal_avg
from sklearn.decomposition import PCA
import numpy as np
from scipy import stats
from scipy.stats import permutation_test
import mne

# magic variables determine the data that I want to look and what type of processing
test_subject = 'IR95'
test_session = 'sess-3'
task = 'wcst'
bp = 1000
event_lock = 'Feedback'
# event_lock = 'Onset'
feature = 'correct'
# feature = 'rule dimension'
standardized_data = True
baseline = (2, 2.5)
# standardized_data = False
epochs_dataset, trial_time, electrode_names, feature_values = lfp_prep(test_subject, test_session,
                                                                       task, event_lock=event_lock,
                                                                       feature=feature,
                                                                       baseline=baseline,
                                                                       electrode_selection='microwire')
organized_data_mean, organized_data, feedback_dict = organize_data(epochs_dataset, feature_values,
                                                                   standardized_data=standardized_data)

n_max_trials, n_electrodes, n_cond, n_timepoints = organized_data.shape
sampling_rate = epochs_dataset.info['sfreq']
mne_info = mne.create_info(electrode_names, sampling_rate, ch_types='seeg')
tfr_power_dict = {}
for i in feedback_dict.keys():
    curr_option = feedback_dict[i]
    trial_num = len(feature_values[feature_values == curr_option])

    data_for_morlet = organized_data[0:trial_num, :, i, :]


    epochs_object_cond = mne.EpochsArray(data_for_morlet, mne_info, tmin=epochs_dataset.tmin)
    tfr_power_dict[curr_option] = morlet(test_subject, test_session, task, epochs_object_cond)

tfr_power_dict['contrast'] = tfr_power_dict['correct'].__sub__(tfr_power_dict['incorrect'])
import matplotlib.pyplot as plt
vmin = -4
vmax = 4
for j in range(n_electrodes):
    fig, axs = plt.subplots(1, 4, figsize=(15, 7), gridspec_kw={'width_ratios': [8, 8, 8, 1]})
    count = 0
    cmap = 'coolwarm'
    for i in tfr_power_dict.keys():
        # ax2 = axs[1, count]
        # ax2.axis('off')
        ax = axs[count]
        mode = 'zscore'
        cax_image = tfr_power_dict[i].plot(
                [j],
                baseline=(2.3, 2.5),
                mode=mode,
                vlim=(vmin, vmax),
                axes=ax,
                show=False,
                colorbar=False, cmap=cmap
            )
        ax.set_title(f"{i}")
        ax.set_xticks([0, 0.3, 0.6, 1, 1.5, 2])
        ax.set_yticks([4, 8, 12, 30, 50, 70, 100, 130, 150, 180])
        count += 1
    # axs[count].axis('off')
    plt.suptitle(f"{electrode_names[j]} - {test_subject} - {test_session}")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, cax=axs[count])
    plt.tight_layout()
    plt.show()


mne_info = mne.create_info(electrode_names, sampling_rate, ch_types='seeg')
