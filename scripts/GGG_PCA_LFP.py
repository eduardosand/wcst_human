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

# We'd like to do a permutation test over each timepoint



# mean_data_p = pca.fit_transform(organized_data_mean.reshape(organized_data_mean.shape[0], -1))
# what is Xa.T? What is dimensionality?

# mean_data_p = pca.fit_transform(Xa.T).T
print(feature_dict)

