from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from GG_dPCA_LFP import lfp_prep, organize_data
from G_dPCA import plot_signal_avg
import scipy
from scipy import stats
# This code exists to run an LDA sweeping timepoint by timepoint

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
epochs_dataset, trial_time, microwire_names, feature_values = lfp_prep(test_subject, test_session,
                                                                       task, event_lock=event_lock,
                                                                       feature=feature,
                                                                       baseline=(2, 2.5), smooth=True,
                                                                       electrode_selection=electrode_selection)

organized_data_mean, organized_data, feedback_dict = organize_data(epochs_dataset, feature_values,
                                                                   standardized_data=standardized_data)

# get rid of the baseline period, NOTE this assumes that we're using feedback locked time periods, and ITI, which will
# change for other analyses
n_neurons, n_cond, n_timepoints = organized_data_mean.shape
feature_dict = feedback_dict
print('plotting now')
plot_signal_avg(organized_data_mean, test_subject, test_session, trial_time, labels=feedback_dict,
                extra_string=f'Normalization = {standardized_data} {event_lock}-lock', signal_names=microwire_names)

mean_accuracies = []
sem_accuracies = []
p_values = []
for i in range(n_timepoints):
    X = epochs_dataset.get_data(copy=False)[:, :, i]
    y = list(feature_values)
    # Assuming X and y are your features and labels
    lda = LinearDiscriminantAnalysis()

    # Perform k-fold cross-validation
    k = 5
    cv = StratifiedKFold(n_splits=k)
    accuracies = cross_val_score(lda, X, y, cv=cv, scoring='accuracy')

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
import matplotlib.pyplot as plt
binsize = 0.5
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
extra_string = f'{extra_string_start} {event_lock}-lock \n corrected for multiple comparisons \n no common average reference'
plt.suptitle(f'Mean Accuracy of LDA on {electrode_selection} broadband LFP \n {test_subject} - session {test_session} \n {extra_string}')
plt.subplots_adjust(hspace=0.7, wspace=0.25, top=0.82, right=right_margin, left=0.1)

# we're plotting the same thing in each subplot, so only grab labels for one plot
# lines, labels = ax.get_legend_handles_labels()
# fig.legend(lines, labels, loc='right', ncol=1)

ax_curr.plot(trial_time, mean_accuracies)
if pvalues_corr is not None:
    for t in range(len(p_values)):
        if pvalues_corr[t] < 0.05:
            ax_curr.axvspan(trial_time[t] - time_diff[t - 1] / 2, trial_time[t] + time_diff[t] / 2, color='red',
                    alpha=0.3)
ax_curr.axvspan(0.3, 0.6, color='grey', alpha=0.3)
ax_curr.axvline(0, linestyle='--', c='black')
# ax_curr.set_title(signal_names[ind])
ax_curr.set_xlabel('')
ax_curr.set_xticks([])

ax_curr.set_xlabel('Time (s)')
ax_curr.set_xticks(time_ticks, time_tick_labels)

## plt.suptitle(f'Mean Activity of SU \n {subject} - session {session} \n {extra_string}')
#plt.subplots_adjust(hspace=0.7, wspace=0.25, top=0.82, right=right_margin, left=0.1)

## we're plotting the same thing in each subplot, so only grab labels for one plot
# lines, labels = ax.get_legend_handles_labels()
# fig.legend(lines, labels, loc='right', ncol=1)

plt.show()
