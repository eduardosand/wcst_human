import statsmodels.api as sm
from pathlib import Path
import os
from behavior_analysis import process_wcst_behavior
import pandas as pd
from intracranial_ephys_utils.load_data import read_file
from scipy.io import loadmat
import numpy as np
from C_su_raster_plots import get_trial_wise_spike_times, plot_neural_spike_trains
import matplotlib.pyplot as plt
# Load modules and data
import statsmodels.api as sm

data = sm.datasets.scotland.load()

# data.exog = sm.add_constant(data.exog)

# Instantiate a gamma family model with the default link function.
# gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())

# gamma_results = gamma_model.fit()

# print(gamma_results.summary())



# Get our spikes
session = 'sess-3'
subject = 'IR95'
data_directory = Path(f"{os.pardir}/data/{subject}/{session}/")
ph_file_path = Path("raw/Events.nev")
beh_directory = data_directory / "behavior"
running_avg = 5
beh_data, rule_shifts_ind, _ = process_wcst_behavior(beh_directory / f"IR95_wcst6_2020_Aug_20_1802.csv",
                                                     running_avg=running_avg)
beh_data.set_index(['trial'], inplace=True)
timestamps_file = f"sub-{subject}-{session}-ph_timestamps.csv"
beh_timestamps = pd.read_csv(beh_directory / timestamps_file)
feedback_times = beh_timestamps['Feedback (seconds)']
onset_times = beh_timestamps['Onset (seconds)']

# matplotlib.rcParams['font.size'] = 12.0

# global t-start
reader = read_file(data_directory / ph_file_path)
reader.parse_header()
start_record = reader.global_t_start
number_spikes = []
curr_neuron = 0
event_lock = 'Feedback'
su_data_dir = data_directory / "sorted/sort/final"
all_su_files = os.listdir(su_data_dir)
for file in all_su_files:
    # step 1. Load in one file, and comb through it for neurons
    microwire_spikes = loadmat(su_data_dir / file)
    neuron_counts = microwire_spikes['useNegative'][0].shape[0]
    for neuron_ind in range(neuron_counts):
        su_cluster_num = microwire_spikes['useNegative'][0][neuron_ind]
        # if su_cluster_num not in [8192, 8488, 14804, 15197]:
        #     continue
        # Note that these timestamps are in microseconds, and according to machine clock
        microsec_sec_trans = 10**-6
        su_timestamps = np.array([[microwire_spikes['newTimestampsNegative'][0, i]*microsec_sec_trans-start_record] for i in
                                  range(microwire_spikes['newTimestampsNegative'].shape[1])
                                 if microwire_spikes['assignedNegative'][0, i] == su_cluster_num])

        tmax = 2.5
        trial_wise_feedback_spikes = get_trial_wise_spike_times(su_timestamps, feedback_times, tmin=-1., tmax=tmax)
        # Plot response in spikes of this one neuron relative to each onset event
        tmin_onset = -0.5
        trial_wise_onset_spikes = get_trial_wise_spike_times(su_timestamps, onset_times, tmin=tmin_onset, tmax=tmax)
        step = 0.1
        trial_time = np.round(np.arange(tmin_onset+step, tmax, step), 2)
        # the big thing from preventing us from getting good code is the design matrix
        spike_counts = np.zeros((len(feedback_times), trial_time.shape[0]))
        if event_lock == 'Feedback':
            trial_wise_spikes = trial_wise_feedback_spikes
        else:
            trial_wise_spikes = trial_wise_onset_spikes
        for trial_ind in range(spike_counts.shape[0]):
            if len(trial_wise_spikes[trial_ind]) == 0:
                continue
            for ind, timestep in enumerate(trial_time):
                # trial_wise_feedback_spikes[trial_ind]
                spike_counts[trial_ind, ind] = len([spike_time for spike_time in trial_wise_spikes[trial_ind]
                                                    if spike_time > timestep and spike_time <= timestep+step])

        spikes_binned = spike_counts
        beh_data.rename(columns={'correct': 'Feedback'}, inplace=True)
        beh_data['Color Rule'] = [1 if i=='Color' else 0 for i in beh_data['rule dimension']]
        beh_data['Texture Rule'] = [1 if i=='Texture' else 0 for i in beh_data['rule dimension']]
        beh_data['Shape Rule'] = [1 if i=='Shape' else 0 for i in beh_data['rule dimension']]
        features = ['Feedback', 'Color Rule', 'Texture Rule', 'Shape Rule']
        X = beh_data[features]
        feature_loadings = np.zeros((4, len(trial_time)))
        p_values = np.ones((4, len(trial_time)))
        for ind, timestep in enumerate(trial_time):
            print(timestep)
            y = spike_counts[:, ind]

            # Don't run the model if there weren't any spikes in that time bin on any trial
            if len(list(set(y))) == 1:
                continue
            print(y)
            # X = epochs_dataset.get_data(copy=False)[:, :, i]

            ### This is super-easy if we rely on built-in GLM fitting code
            glm_poisson_exp = sm.GLM(y, X, family=sm.families.Poisson())
            results = glm_poisson_exp.fit()
            feature_loadings[:, ind] = results.params
            p_values[:, ind] = results.pvalues
            print(results.summary())

        # pGLM_results = glm_poisson_exp.fit(max_iter=100, tol=1e-6, tol_criterion='params')
        print('huh')
        fig, ax = plt.subplots(figsize=(8, 2))

        binsize = 0.2
        tmin = tmin_onset
        tmax_actual = tmax
        trial_time = np.round(np.arange(tmin+step, tmax_actual, step), 2)
        min_multiple = np.min(trial_time)
        time_ticks = np.arange(min_multiple, np.max(trial_time) + binsize, step=binsize)
        time_ticks = [round(i,1) for i in time_ticks]
        time_tick_labels = time_ticks
        time_tick_labels = [f'{i:.1f}' for i in time_tick_labels]
        time_diff = np.diff(trial_time)
        # find minimum of minima & maximum of maxima
        # feature_loadings[p_values > 0.05] = 0
        vmin = np.min(feature_loadings)
        vmax = np.max(feature_loadings)
        alpha = np.ones_like(feature_loadings)
        alpha[p_values > 0.05] = 0.2
        # feature_loadings[p_values > 0.05] = vmax - (vmax-vmin)/2
        im = ax.imshow(feature_loadings, vmin=vmin, vmax=vmax, cmap='viridis', alpha=alpha)

        # Show all ticks and label them with the respective list entries
        lda_time_ticks = np.array([ind for ind, i in enumerate(trial_time) if round(i,1) in time_ticks])
        ax.set_xticks(lda_time_ticks, time_tick_labels)
        ax.set_yticks(np.arange(len(features)), labels=features, fontsize=8)
        ax.set_title(f'Poisson Regression Weights on Cluster {su_cluster_num}, {event_lock}-lock')
        ax.set_ylabel("Feature Regressors")
        ax.set_xlabel("Time (s)")
        # ax.set_title(f'LDA weights on {electrode_selection} electrodes, {event_lock}-lock')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # add space for colour bar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.3, 0.04, 0.5])
        fig.colorbar(im, cax=cbar_ax)
        # plt.tight_layout()
        plt.savefig(f'{os.pardir}/results/Pois_reg_{subject}_{session}_{su_cluster_num}_{step}_{event_lock}_loadings.svg')

        plt.show()
        # plt.savefig(f'{os.pardir}/results/LDA_{test_subject}_{test_session}_{electrode_selection}_loadings.svg')

        # from matplotlib import pyplot as plt
        # sort_order = sorted(set(beh_data['Feedback']))
        # if len(sort_order) == 3:
        #     color_dict = dict(zip(sort_order, ['red', 'green', 'blue']))
        # else:
        #     color_dict = dict(zip(sort_order, ['purple', 'orange']))
        # fig, ax = plt.subplots()
        # plot_neural_spike_trains(ax, trial_wise_feedback_spikes, beh_data['Feedback'], color_dict)

        # pGLM_const = glm_poisson_exp[-1].fit_['beta0'] # constant ("dc term)")
        # pGLM_const = pGLM_results.params[0]
        # pGLM_filt = pGLM_results.params[1:] # stimulus filter

        # The 'GLM' function can fit a GLM for us. Here we have specified that
        # we want the noise model to be Poisson. The default setting for the link
        # function (the inverse of the nonlinearity) is 'log', so default
        # nonlinearity is 'exp').

        ### Compute predicted spike rate on training data
        # rate_pred_pGLM = np.exp(pGLM_const + design_mat @ pGLM_filt)