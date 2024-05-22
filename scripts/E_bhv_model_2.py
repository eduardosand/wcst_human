import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
from bhv_data_convert import bhv_convert
from behavior_analysis import process_wcst_behavior
import matplotlib.patches as patches

mpl.rcParams['agg.path.chunksize'] = 10000

# Essentially the problem we have is that our trained models spit out feature probabilities, but we might use all 3
# features to choose an object
# This script from Vishwa's repo is used to train a classifier on feature probabilities to get object probabilities
# It is modified here so we can get the following

# RPE - we'll get the object probability and then subtract feedback signals

# Bayesian surprise - For this we'll use the feature probabilities and calculate the KL divergence between the feature
# probabilities before and after feedback (this is harder to compute)

from pathlib import Path
import os
import pickle
species = 'human'
subj = 'b01'
dataDirectory = Path(f'{os.pardir}/{os.pardir}/WCST_behavioral_model_and_analysis/rawData/history/{species}')

# three of the parameters are solely related to the features of the input data
# which are presaved
glmLag = 1
intercept = 1
# dataName = Path(f'{subj}_{glmLag}_super.pickle')
#
# # Load training data
# with open(dataDirectory / dataName, 'rb') as f:
#     [_, _, ruleSuperBlocks, chosenObjectSuperBlocks, stimulusSuperBlocks] = pickle.load(f)
#
# # ruleSuperBlocks contains the rule for each trial and each session
# # the coding is by number, we can do this
# print(ruleSuperBlocks)

# chosenObjectSuperBlocks contains which object they chose, it's gonna be one through 4

# Okay we more or less know what's in super, we have to build the right objects now so that we can get
# the object prediction code to work
# Basically we want to create a version of the analysis code that Vishwa has that works for us
subject = 'IR95'
session = 'sess-3'
kfoldnum = 0
glmType = 1  # Bernoulli
lag = 1
num_states = 4
observation_noise = 0
diagonal_p = 0.9
wzero = 1
block_method = 0
save_directory = Path(f"{os.pardir}/data/{subject}/{session}/model")
save_name = (f'glmtype{glmType}_lag{glmLag}_intercept{intercept}_kfold{kfoldnum}_numstates{num_states}_observationnoise'
             f'{observation_noise}_diagonalp{diagonal_p}_wzero{wzero}.pickle')

with open(save_directory / save_name, 'rb') as f:
    try:
        [glmhmm, fit_ll, train_ll, test_ll, trCnt, teCnt] = pickle.load(f)
    except:
        [glmhmm, fit_ll, train_ll, test_ll] = pickle.load(f)

history_data, choice_data = bhv_convert(subject, session, intercept)

# Lucky for me there is just one block

viterbiStates = np.array([glmhmm.most_likely_states(choice_data[:, f, :].astype('int'),
                                                    input=history_data[:, f, :]) for f in range(12)])

data_directory = Path(f"{os.pardir}/data/{subject}/{session}/")
beh_directory = data_directory / "behavior"

# Will Need to be changed
beh_data, rule_shifts_ind, _ = process_wcst_behavior(beh_directory / f"{subject}_wcst6_2020_Aug_20_1802.csv")
beh_data.set_index(['trial'], inplace=True)


features = ['S', 'T', 'C', 'Q', 'B', 'Y', 'G', 'M', 'L', 'P', 'W', 'R']
feature_dict = dict(zip(features, np.arange(12)))
# Use dictionary to get numbers of rules
rule = beh_data['rule'].map(feature_dict)

# define color map for states
color_map = {0: np.array([215, 48, 39]),
             1: np.array([252, 141, 89]),
             2: np.array([166, 206, 227]),
             3: np.array([66, 146, 198])}

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4*2*5, 4*2*3))
# make a 3d numpy array that has a color channel dimension
data_3d = np.ndarray(shape=(viterbiStates.shape[0], viterbiStates.shape[1], 3), dtype=int)
for i in range(0, viterbiStates.shape[0]):
    for j in range(0, viterbiStates.shape[1]):
        data_3d[i][j] = color_map[viterbiStates[i][j]]

ax.imshow(data_3d, extent=[0, 200, 0, 12], aspect=10)
st = 0
ln = 1
n_trials = viterbiStates.shape[1]
cRule = list(rule)[0]
for i in range(1, n_trials):
    if cRule == list(rule)[i]:
        ln += 1
    else:
        rect = patches.Rectangle((st, 11 - cRule), ln, 1, linewidth=3, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
        st = i
        ln = 1
        cRule = list(rule)[i]

# Prettify
    ax.plot([0, 200], [8,8],linewidth=4,color='#6a3d9a')
    ax.plot([0, 200], [4,4],linewidth=4,color='#6a3d9a')
    ax.tick_params(axis='both',direction='out',labelsize= 15, width=2, length=12)
    # ax.tick_params(axis='x',length=5,labelsize=18)
    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_axisbelow(True)
    # ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # plt.savefig('exampleMonkeyStates.eps', format='eps')
    ax.set_title(f'HMM Model for {subject} , {session}, \n kfold {kfoldnum}, '
                 f'number of states = {num_states}', fontsize=60)

plt.show()
print(viterbiStates)