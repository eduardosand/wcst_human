from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from BB_processlfp import lfp_prep, organize_data, plot_signal_avg
import scipy
from scipy import stats
import os
import pandas as pd
# This code exists to run an LDA sweeping timepoint by timepoint

def lda_analysis(organized_data_mean, organized_data, trial_time, test_subject, test_session, signal_names,
                 electrode_selection, feature_values, event_lock, decomposition, feedback_dict):
    n_neurons, n_cond, n_timepoints = organized_data_mean.shape
    mean_accuracies = []
    sem_accuracies = []
    p_values = []
    lda_loadings = []
    for i in range(n_timepoints):
        X = organized_data[:, :, i]
        # X = epochs_dataset.get_data(copy=False)[:, :, i]
        y = list(feature_values)
        # Assuming X and y are your features and labels
        lda = LinearDiscriminantAnalysis()

        # Perform k-fold cross-validation
        k = 5
        cv = StratifiedKFold(n_splits=k)
        accuracies = cross_val_score(lda, X, y, cv=cv, scoring='accuracy')
        lda.fit(X,y)
        # get coefficients
        lda_loadings.append(lda.coef_)
        # Average accuracy
        mean_accuracy = np.mean(accuracies)
        sem = scipy.stats.sem(accuracies)
        mean_accuracies.append(mean_accuracy)
        sem_accuracies.append(sem)
        print(f"Mean Accuracy: {mean_accuracy} for timepoint {trial_time[i]}")

        # Permutation testing
        n_permutations = 1000
        perm_accuracies = []

        for _ in range(n_permutations):
            y_permuted = np.random.permutation(y)
            perm_acc = cross_val_score(lda, X, y_permuted, cv=cv, scoring='accuracy')
            perm_accuracies.append(np.mean(perm_acc))

        # Calculate p-value
        perm_accuracies = np.array(perm_accuracies)
        p_value = np.mean(perm_accuracies >= mean_accuracy)
        p_values.append(p_value)
        print(p_values)

    pvalues_corr = stats.false_discovery_control(p_values, method='bh')
    print(pvalues_corr)
    print(trial_time[pvalues_corr < 0.05])
    print(np.max(mean_accuracies))
    import matplotlib.pyplot as plt
    binsize = 0.25
    min_multiple = np.min(trial_time) // binsize
    time_ticks = np.arange(min_multiple * binsize, np.max(trial_time) + binsize, step=binsize)
    time_tick_labels = time_ticks
    time_tick_labels = [f'{i:.1f}' for i in time_tick_labels]
    time_diff = np.diff(trial_time)
    n_electrodes, n_cond, n_timepoints = organized_data_mean.shape
    labels = feedback_dict

    # Calculate the maximum label length to adjust the right margin dynamically
    max_label_length = max(len(str(label)) for label in labels.values())
    right_margin = 0.7 + max_label_length * 0.01

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharex=True)
    ax_curr = ax
    ax_curr.set_xlabel('Time (s)')
    ax_curr.set_xticks(time_ticks, time_tick_labels)
    extra_string_start = 'normalized' + ', bp and notch filtered'
    # if car_setting:
    #     extra_string = f'{extra_string_start} {event_lock}-lock \n corrected for multiple comparisons \n CAR'
    # else:
    extra_string = f'{extra_string_start} {event_lock}-lock \n corrected for multiple comparisons'
    # plt.suptitle(f'Mean Accuracy of LDA on {electrode_selection} broadband LFP \n {test_subject} - session {test_session} '
    #              f'\n {extra_string}')
    if electrode_selection == 'all':
        plt.suptitle(f'LDA Accuracy on \n {electrode_selection} broadband LFP', fontsize=26)
    else:
        plt.suptitle(f'LDA Accuracy on \n {electrode_selection} broadband LFP', fontsize=26)
    plt.subplots_adjust(hspace=0.7, wspace=0.25, top=0.82, right=right_margin, left=0.1)

    # we're plotting the same thing in each subplot, so only grab labels for one plot
    # lines, labels = ax.get_legend_handles_labels()
    # fig.legend(lines, labels, loc='right', ncol=1)

    ax_curr.plot(trial_time, mean_accuracies)
    if pvalues_corr is not None:
        for t in range(len(p_values)):
            if pvalues_corr[t] < 0.05 and t == len(p_values)-1:
                ax_curr.axvspan(trial_time[t] - time_diff[t - 1] / 2, trial_time[t], color='red',
                        alpha=0.3)
            elif pvalues_corr[t] < 0.05 and t == 0:
                ax_curr.axvspan(trial_time[t], trial_time[t] + time_diff[t] / 2, color='red',
                        alpha=0.3)
            elif pvalues_corr[t] < 0.05:
                ax_curr.axvspan(trial_time[t] - time_diff[t - 1] / 2, trial_time[t] + time_diff[t] / 2, color='red',
                        alpha=0.3)
    # ax_curr.axvspan(0.3, 0.6, color='grey', alpha=0.3)
    ax_curr.axvline(0, linestyle='--', c='black')
    chance_performance = np.ones((len(trial_time),1)) * 0.5
    ax_curr.plot(trial_time, chance_performance, linestyle='--', color='purple')
    # ax_curr.set_title(signal_names[ind])

    ax_curr.set_xlabel('Time (s)', fontsize=20)
    ax_curr.set_ylabel('Decoding Accuracy', fontsize=20)
    ax_curr.set_xticks(time_ticks, time_tick_labels)
    ax_curr.set_xlim([np.min(trial_time), np.max(trial_time)])
    ## plt.suptitle(f'Mean Activity of SU \n {subject} - session {session} \n {extra_string}')
    #plt.subplots_adjust(hspace=0.7, wspace=0.25, top=0.82, right=right_margin, left=0.1)

    ## we're plotting the same thing in each subplot, so only grab labels for one plot
    # lines, labels = ax.get_legend_handles_labels()
    # fig.legend(lines, labels, loc='right', ncol=1)
    plt.savefig(f'{os.pardir}/results/LDA_{test_subject}_{test_session}_{electrode_selection}.svg')
    plt.show()
    np.savez(f'{os.pardir}/results/LDA_{test_subject}_{test_session}_{electrode_selection}_loadings.npz',
             np.array(lda_loadings), trial_time, signal_names)

    lda_loadings = np.squeeze(np.array(lda_loadings)).T
    if electrode_selection == 'all':
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_ylabel('Electrode Locations', fontsize=18)
        label_textsize = 6
    else:
        fig, ax = plt.subplots(figsize=(7,4))
        ax.set_ylabel('Electrode \n Locations', fontsize=18)
        label_textsize = 8

    # find minimum of minima & maximum of maxima
    vmin = np.min(lda_loadings)
    vmax = np.max(lda_loadings)
    im = ax.imshow(lda_loadings, vmin=vmin, vmax=vmax, cmap='viridis')

    # Show all ticks and label them with the respective list entries
    lda_time_ticks = np.array([ind for ind,i in enumerate(trial_time) if i in time_ticks])
    ax.set_xticks(lda_time_ticks, time_tick_labels[1:-1])

    anat_csv = f"{os.pardir}/data/{test_subject}/{test_subject}_Elec_Notes.xlsx"
    # anat_csv = "/path/to/csv/file/with/mni/coordinates/coords_mni.csv"
    # raw = read_raw_fif("/path/to/seeg/sub-009_preproc_ieeg.fif")  # preprocessed seeg data

    # epoch at start of audio playback
    # epochs = mne.Epochs(raw, event_id={"wav_playback"}, detrend=1, baseline=None)

    # load the coordinate values
    coords = pd.read_excel(anat_csv)
    # coords = coords[["Electrode", "MNI_1", "MNI_2", "MNI_3", "Loc Meeting"]]

    localization_dict = {}
    feature_key_values = [(coords.loc[ind, "Electrode"], coords.loc[ind, "Loc Meeting"]) for ind in coords.index.values]
    localization_dict.update(feature_key_values)
    localization_names = [localization_dict[wire] if not wire.startswith('m') else 'u' +
                                                                                   localization_dict[wire[1:4] +'1']
                          for ind, wire in enumerate(signal_names)]

    ax.set_yticks(np.arange(len(signal_names)), labels=localization_names, fontsize=label_textsize)

    ax.set_xlabel('Time(s)', fontsize=18)

    # ax.set_title(f'LDA weights on {electrode_selection} electrodes, {event_lock}-lock')
    ax.set_title(f'LDA weights on \n {electrode_selection} electrodes', fontsize=26)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.tight_layout()

    # add space for colour bar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.04, 0.5])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(f'{os.pardir}/results/LDA_{test_subject}_{test_session}_{electrode_selection}_{decomposition}_loadings.svg')
    plt.show()


def main():
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
    # standardized_data = False
    electrode_selection = 'all'
    decomposition = 'broadband'
    # baseline=(2,2.5)
    baseline = (-0.5, 0)
    car_setting = False
    epochs_dataset, trial_time, signal_names, feature_values = lfp_prep(test_subject, test_session,
                                                                           task, event_lock=event_lock,
                                                                           feature=feature,
                                                                           baseline=baseline, smooth=True,
                                                                           electrode_selection=electrode_selection,
                                                                           car=car_setting)

    organized_data_mean, organized_data, feedback_dict = organize_data(epochs_dataset, feature_values,
                                                                       standardized_data=standardized_data, method='LDA')

    # get rid of the baseline period, NOTE this assumes that we're using feedback locked time periods, and ITI, which will
    # change for other analyses
    feature_dict = feedback_dict
    # print('plotting now')
    extra_string_plot = f'{event_lock} - lock'
    if standardized_data:
        extra_string_plot = extra_string_plot + ', normalized'
    if car_setting:
        extra_string_plot = extra_string_plot + ', car'


    plot_signal_avg(organized_data_mean, test_subject, test_session, trial_time, labels=feedback_dict,
                    extra_string=extra_string_plot, signal_names=signal_names)

    lda_analysis(organized_data_mean, organized_data, trial_time, test_subject, test_session,
                 signal_names, electrode_selection, feature_values, event_lock, decomposition, feedback_dict)

if __name__ == "__main__":
    main()