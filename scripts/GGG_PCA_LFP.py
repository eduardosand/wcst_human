# This code exists to perform a PCA analysis of the broadband LFP data
from GG_dPCA_LFP import lfp_prep, organize_data
from G_dPCA import plot_signal_avg
from sklearn.decomposition import PCA
import numpy as np
from scipy import stats
from scipy.stats import permutation_test

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
epochs_dataset, trial_time, microwire_names, feature_values = lfp_prep(test_subject, test_session,
                                                                       task, event_lock=event_lock,
                                                                       feature=feature,
                                                                       baseline=(2, 2.5), smooth=True)

organized_data_mean, organized_data, feedback_dict = organize_data(epochs_dataset, feature_values,
                                                                   standardized_data=standardized_data, method='PCA')

# get rid of the baseline period, NOTE this assumes that we're using feedback locked time periods, and ITI, which will
# change for other analyses
n_neurons, n_cond, n_timepoints = organized_data_mean.shape
feature_dict = feedback_dict
print('plotting now')
plot_signal_avg(organized_data_mean, test_subject, test_session, trial_time, labels=feedback_dict,
                extra_string=f'Normalization = {standardized_data} {event_lock}-lock', signal_names=microwire_names)

pca = PCA()
# we're using trial-averaged PCA with multiple conditions
# we want to fit on data that is n_samples X n_features, corresponding to (n_timepoints * n_cond) X n_neurons for us
pca_trialwise = PCA()
mean_data_pc = pca.fit_transform(organized_data_mean.reshape(organized_data_mean.shape[0], -1).T).T
pc_data_mean = mean_data_pc.reshape(organized_data_mean.shape)
pc_names = [f'PC_{i+1}: {pca.explained_variance_ratio_[i]:.2%}' for i in np.arange(pc_data_mean.shape[0])]
plot_signal_avg(pc_data_mean, test_subject, test_session, trial_time, labels=feedback_dict,
                extra_string=f'Normalization = {standardized_data} {event_lock}-lock', signal_names=pc_names)

data_pc = pca_trialwise.fit_transform(organized_data).T
trial_wise_loadings = data_pc.reshape(n_neurons, len(feature_values), n_timepoints)
mean_loadings = np.zeros((n_neurons, len(set(feature_values)), n_timepoints))
for i, key in enumerate(feature_dict.keys()):
    mean_loadings[:, i, :] = np.mean(trial_wise_loadings[:, feature_values == feature_dict[key], :], axis=1)
trialwise_pc_names = [f'average trialwise_PC_{i+1}: {pca_trialwise.explained_variance_ratio_[i]:.2%}' for i in
                      np.arange(data_pc.shape[0])]


def mean_diff_statistic(x, y, axis):
    return np.abs(np.mean(x, axis=axis) - np.mean(y, axis=axis))

# because our statistic is vectorized, we pass `vectorized=True`
# We'd like to do a permutation test over each timepoint
# No way to make this cleanly, but with two conditions for features, it's straightforward
rng = np.random.default_rng()
# only check on the first 10 components and don't check baseline period
n_pc_analyzed = 7
n_timepoints_check = int(len(trial_time[(trial_time < 1.5) & (trial_time > 0)]))
starting_index = np.argmax(trial_time > 0)
# n_timepoints_check = int(sum(trial_time < 1.5))
pvalues_uncorr = np.ones((n_pc_analyzed, n_timepoints_check))
for i in range(n_pc_analyzed):
    for j in range(n_timepoints_check):
        new_j = j+starting_index
        res = permutation_test((trial_wise_loadings[i, feature_values == feature_dict[0], new_j],
                                trial_wise_loadings[i, feature_values == feature_dict[1], new_j]), mean_diff_statistic,
                                vectorized=True,
                                n_resamples=99999, alternative='greater', random_state=rng)
        pvalues_uncorr[i, j] = res.pvalue


pvalues_corr = stats.false_discovery_control(pvalues_uncorr.flatten(), method='bh')
pvalues_corr = pvalues_corr.reshape(n_pc_analyzed, n_timepoints_check)

pvalues_final = np.ones((n_pc_analyzed, n_timepoints))
starting_index = np.argmax(trial_time > 0)
pvalues_final[:, starting_index:n_timepoints_check+starting_index] = pvalues_uncorr

extra_string_start = 'normalized' if standardized_data else 'unnormalized'
extra_string_start = extra_string_start + ', bp and notch filtered'

plot_signal_avg(mean_loadings[:n_pc_analyzed, :, :], test_subject, test_session, trial_time, labels=feedback_dict,
                extra_string=f'{extra_string_start} {event_lock}-lock', signal_names=trialwise_pc_names,
                pvalues=pvalues_final)


pvalues_final[:n_pc_analyzed, starting_index:n_timepoints_check+starting_index] = pvalues_corr
plot_signal_avg(mean_loadings[:n_pc_analyzed, :, :], test_subject, test_session, trial_time, labels=feedback_dict,
                extra_string=f'{extra_string_start} {event_lock}-lock \n '
                             f'corrected for multiple comparisons', signal_names=trialwise_pc_names,
                pvalues=pvalues_final)

print(feature_dict)

