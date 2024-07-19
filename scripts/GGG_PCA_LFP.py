# This code exists to perform a PCA analysis of the broadband LFP data
from GG_dPCA_LFP import lfp_prep
from G_dPCA import plot_signal_avg
from sklearn.decomposition import PCA
import numpy as np

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
organized_data_mean, organized_data, feedback_dict, trial_time, microwire_names, feature_values = lfp_prep(test_subject, test_session,
                                                                                           task, event_lock=event_lock,
                                                                                           feature=feature,
                                                                                           standardized_data=standardized_data)

# get rid of the baseline period, NOTE this assumes that we're using feedback locked time periods, and ITI, which will
# change for other analyses
n_neurons, n_cond, n_timepoints = organized_data_mean.shape
feature_dict = feedback_dict

# suptitle=f'All microwires, bandpassed at {bp}, {lock}-locked'
plot_signal_avg(organized_data_mean, test_subject, test_session, trial_time, labels=feedback_dict,
                extra_string=f'Normalization = {standardized_data} {event_lock}-lock', signal_names=microwire_names)

pca = PCA()
# we're using trial-averaged PCA with multiple conditions
# we want to fit on data that is n_samples X n_features, corresponding to (n_timepoints * n_cond) X n_neurons for us

# a[~np.isnan(a).any(axis=1), :]
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

plot_signal_avg(mean_loadings, test_subject, test_session, trial_time, labels=feedback_dict,
                extra_string=f'Normalization = {standardized_data} {event_lock}-lock', signal_names=trialwise_pc_names)


def mean_diff_statistic(x, y, axis):
    return np.abs(np.mean(x, axis=axis) - np.mean(y, axis=axis))


from scipy.stats import permutation_test
# because our statistic is vectorized, we pass `vectorized=True`
# `n_resamples=np.inf` indicates that an exact test is to be performed
# We'd like to do a permutation test over each timepoint
# No way to make this cleanly, but with two conditions for features, it's straightforward
rng = np.random.default_rng()
# only check on the first 10 components
n_pc_analyzed = 10
# don't check baseline
n_timepoints_check = int(sum(trial_time < 2.))

pvalues_uncorr = np.ones((n_pc_analyzed, 1, n_timepoints_check))
for i in range(n_pc_analyzed):
    for j in range(n_timepoints_check):
        res = permutation_test((trial_wise_loadings[i, feature_values == feature_dict[0], j],
                                trial_wise_loadings[i, feature_values == feature_dict[1], j]), mean_diff_statistic,
                                vectorized=True,
                                n_resamples=99999, alternative='greater', random_state=rng)
        pvalues_uncorr[i, 0, j] = res.pvalue

print(np.sum(pvalues_uncorr.flatten() < 0.05))
import scipy.stats
pvalues_corr = scipy.stats.false_discovery_control(pvalues_uncorr.flatten(), method='bh')
print(np.sum(pvalues_corr < 0.05))
pvalues_corr = pvalues_corr.reshape(n_pc_analyzed, 1, n_timepoints_check)

pvalues_final = np.ones((n_neurons,1,n_timepoints))
pvalues_final[:n_pc_analyzed, 0, :n_timepoints_check] = pvalues_corr
# mean_data_p = pca.fit_transform(organized_data_mean.reshape(organized_data_mean.shape[0], -1))
# what is Xa.T? What is dimensionality?

# mean_data_p = pca.fit_transform(Xa.T).T
print(feature_dict)

