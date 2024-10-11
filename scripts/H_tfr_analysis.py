# This code exists to perform a PCA analysis of the broadband LFP data
from BB_processlfp import lfp_prep, organize_data, plot_signal_avg, morlet
import numpy as np
from scipy import stats
from scipy.stats import permutation_test
import mne
import os
import pandas as pd

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
# baseline = (2, 2.5)
baseline = (-0.5, 0)
electrode_selection = 'macrocontact'
# standardized_data = False
epochs_dataset, trial_time, electrode_names, feature_values = lfp_prep(test_subject, test_session,
                                                                       task, event_lock=event_lock,
                                                                       feature=feature,
                                                                       baseline=baseline,
                                                                       electrode_selection=electrode_selection)
organized_data_mean, organized_data, feedback_dict = organize_data(epochs_dataset, feature_values,
                                                                   standardized_data=standardized_data)

anat_csv = f"{os.pardir}/data/{test_subject}/{test_subject}_Elec_Notes.xlsx"
# load the coordinate values
coords = pd.read_excel(anat_csv)
localization_dict = {}
feature_key_values = [(coords.loc[ind, "Electrode"], coords.loc[ind, "Loc Meeting"]) for ind in coords.index.values]
localization_dict.update(feature_key_values)
localization_names = [localization_dict[wire] if not wire.startswith('m') else 'u' +
                                                                               localization_dict[wire[1:4] +'1']
                      for ind, wire in enumerate(electrode_names)]
n_neurons, n_cond, n_timepoints = organized_data_mean.shape
feature_dict = feedback_dict
# print(feature_dict)
# print('plotting now')
plot_signal_avg(organized_data_mean, test_subject, test_session, trial_time, labels=feedback_dict,
                extra_string=f'Normalization = {standardized_data} {event_lock}-lock', signal_names=electrode_names)

n_max_trials, n_electrodes, n_cond, n_timepoints = organized_data.shape
sampling_rate = epochs_dataset.info['sfreq']
mne_info = mne.create_info(electrode_names, sampling_rate, ch_types='seeg')
tfr_power_dict = {}
trial_num_dict = {}
for i in sorted(feedback_dict.keys()):
    curr_option = feedback_dict[i]
    trial_num = len(feature_values[feature_values == curr_option])

    data_for_morlet = organized_data[0:trial_num, :, i, :]
    epochs_object_cond = mne.EpochsArray(data_for_morlet, mne_info, tmin=epochs_dataset.tmin)
    tfr_power_dict[curr_option] = morlet(test_subject, test_session, task, epochs_object_cond, standardized=True,
                                         baseline=baseline, trial_baseline=True)
    trial_num_dict[curr_option] = trial_num
freqs = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 21, 24, 27, 30, 35,
                  40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 220])
tfr_power_dict['contrast'] = tfr_power_dict['correct'].__sub__(tfr_power_dict['incorrect'])
trial_num_dict['contrast'] = 0
import matplotlib.pyplot as plt
vmin = -1
vmax = 1
for j in range(n_electrodes):
    fig, axs = plt.subplots(1, 4, figsize=(15, 7), gridspec_kw={'width_ratios': [8, 8, 8, 1]})
    count = 0
    cmap = 'coolwarm'
    for i in tfr_power_dict.keys():
        ax = axs[count]
        # fig, ax = plt.subplots()
        mode = 'zscore'
        baseline = None
        cax_image = tfr_power_dict[i].plot(
                [j],
                baseline=baseline,
                mode=mode,
                vlim=(vmin, vmax),
                axes=ax,
                show=False,
                colorbar=False, cmap=cmap, yscale='log',
            )
        if i != 'contrast':
            ax.set_title(f"{i} - n={trial_num_dict[i]}", fontsize=20)
        else:
            ax.set_title(f"{i}", fontsize=20)
        ax.set_xticks([-0.5, 0, 0.3, 0.6, 1, 1.5])
        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim([np.min(freqs), np.max(freqs)])
        ax.set_ylabel("")
        # ax.set_yticks(freqs, freqs)
        # ax.set_yticks([4, 5, 6, 7, 8, 9])
        count += 1
    # axs[count].axis('off')
    # plt.suptitle(f"{electrode_names[j]} - {test_subject} - {test_session}")
    plt.suptitle(f"{localization_names[j]}", fontsize=30)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, cax=axs[count])
    plt.tight_layout()
    plt.savefig(f"{os.pardir}/results/tfr_{electrode_names[j]}_{test_subject}_{test_session}_{feature}.svg")
    plt.savefig(f"{os.pardir}/results/tfr_{electrode_names[j]}_{test_subject}_{test_session}_{feature}.png",
                dpi=600)
    plt.show()


mne_info = mne.create_info(electrode_names, sampling_rate, ch_types='seeg')
