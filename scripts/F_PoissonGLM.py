import statsmodels.api as sm
from pathlib import Path
import os
from behavior_analysis import process_wcst_behavior
import pandas as pd
from intracranial_ephys_utils.load_data import read_file
from scipy.io import loadmat
import numpy as np
from C_su_raster_plots import get_trial_wise_times

# Load modules and data
import statsmodels.api as sm

data = sm.datasets.scotland.load()

data.exog = sm.add_constant(data.exog)

# Instantiate a gamma family model with the default link function.
gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())

gamma_results = gamma_model.fit()

print(gamma_results.summary())



# Get our spikes
session = 'sess-4'
subject = 'IR87'
data_directory = Path(f"{os.pardir}/data/{subject}/{session}/")
ph_file_path = Path("raw/Events.nev")
beh_directory = data_directory / "behavior"
running_avg = 5
beh_data, rule_shifts_ind, _ = process_wcst_behavior(beh_directory / f"sub-{subject}-{session}-beh.csv",
                                                     running_avg=running_avg)
beh_data.set_index(['trial'], inplace=True)
timestamps_file = "sub-IR87-sess-4-ph_timestamps.csv"
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

su_data_dir = data_directory / "sorted/sort/final"
all_su_files = os.listdir(su_data_dir)
for file in all_su_files:
    # step 1. Load in one file, and comb through it for neurons
    microwire_spikes = loadmat(su_data_dir / file)
    neuron_counts = microwire_spikes['useNegative'][0].shape[0]
    for neuron_ind in range(neuron_counts):
        su_cluster_num = microwire_spikes['useNegative'][0][neuron_ind]

        # Note that these timestamps are in microseconds, and according to machine clock
        microsec_sec_trans = 10**-6
        su_timestamps = np.array([[microwire_spikes['newTimestampsNegative'][0, i]*microsec_sec_trans-start_record] for i in
                                  range(microwire_spikes['newTimestampsNegative'].shape[1])
                                 if microwire_spikes['assignedNegative'][0, i] == su_cluster_num])

        trial_wise_feedback_spikes = get_trial_wise_times(su_timestamps, feedback_times, beh_data, tmin=-1., tmax=1.5)
        # Plot response in spikes of this one neuron relative to each onset event
        tmin_onset = -0.5
        trial_wise_onset_spikes = get_trial_wise_times(su_timestamps, onset_times, beh_data, tmin=tmin_onset, tmax=1.5)


        spikes_binned = trial_wise_onset_spikes
        ### This is super-easy if we rely on built-in GLM fitting code
        glm_poisson_exp = sm.GLM(endog=spikes_binned, exog=design_mat_offset,
                                 family=sm.families.Poisson())

        pGLM_results = glm_poisson_exp.fit(max_iter=100, tol=1e-6, tol_criterion='params')


        # pGLM_const = glm_poisson_exp[-1].fit_['beta0'] # constant ("dc term)")
        pGLM_const = pGLM_results.params[0]
        pGLM_filt = pGLM_results.params[1:] # stimulus filter

        # The 'GLM' function can fit a GLM for us. Here we have specified that
        # we want the noise model to be Poisson. The default setting for the link
        # function (the inverse of the nonlinearity) is 'log', so default
        # nonlinearity is 'exp').

        ### Compute predicted spike rate on training data
        rate_pred_pGLM = np.exp(pGLM_const + design_mat @ pGLM_filt)