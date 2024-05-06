from behavior_analysis import process_wcst_behavior
import os
from pathlib import Path
import numpy as np
import ssm
import numpy.random as npr
from bhv_data_convert import bhv_convert
import pickle
from sklearn.model_selection import KFold
# This script is modified from Vishwa Gaudar's git repo for paper.
# Tested may 2024


def initializeObservation(glmhmm,glmhmmOne,sigma):
    """
    function to initialize observation weights of glmhmm
    :param glmhmm:
    :param glmhmmOne:
    :param sigma:
    :return:
    """

    for k in range(glmhmm.K):
        oShape = glmhmmOne.observations.params[0].shape
        glmhmm.observations.params[k] = glmhmmOne.observations.params[0] * (1 + sigma * npr.rand(oShape[0],oShape[1]))

    return glmhmm


def initializeTransition(glmhmm,glmhmmOne,diagonalP,wZero):
    """
    function to initialize transition weights of glmhmmOne
    :param glmhmm:
    :param glmhmmOne:
    :param diagonalP:
    :param wZero:
    :return:
    """

    # initilize log_Ps variable
    Ps = diagonalP * np.eye(glmhmm.K) + .05 * npr.rand(glmhmm.K, glmhmm.K)
    Ps /= Ps.sum(axis=1, keepdims=True)
    glmhmm.transitions.log_Ps = np.log(Ps)

    # initialize Ws variable
    if wZero == 1:
        glmhmm.transitions.Ws = np.zeros(glmhmm.transitions.Ws.shape)
    elif wZero == 2:
        for k1 in range(glmhmm.K):
            for k2 in range(glmhmm.K):
                glmhmm.transitions.Ws[k1,k2,:] = glmhmmOne.observations.params[0]

    return glmhmm


subject = 'IR95'
session = 'sess-3'
intercept = 1
glmType = 1  # Bernoulli
glmLag = 1
num_states = 3
observation_noise = 0
diagonal_p = 0.9
wzero = 1
block_method = 0

history_data, choice_data = bhv_convert(subject, session, intercept)
Xdata = history_data
Ydata = choice_data
save_directory = Path(f"{os.pardir}/data/{subject}/{session}/model")

species = 'human'
subj = 'b01'
dataDirectory = Path(f'{os.pardir}/{os.pardir}/WCST_behavioral_model_and_analysis/rawData/inputs/{species}/{subj}')
# three of the parameters are solely related to the features of the input data
# which are presaved
dataName = Path(f'glm{glmType}_lag{glmLag}_int{intercept}.pickle')

# Load training data
with open(dataDirectory / dataName, 'rb') as f:
    [XData_sample, YData_sample] = pickle.load(f)

# then map the rule to these keys to replace some things with 1
rule_dict = {'S': 'Problem', 'T': 'Shape', 'C': 'Shape', 'Q': 'Shape', 'B': 'Color', 'Y': 'Color', 'G': 'Color',
                     'M': 'Color', 'L': 'Texture', 'P': 'Texture', 'W': 'Texture', 'R': 'Texture'}
features = ['S0', 'T', 'C', 'Q', 'B', 'Y', 'G', 'M', 'L', 'P', 'S2', 'R']
feature_dict = dict(zip(features, np.linspace(0, 12)))
# then create an array but with keys, so we can keep track of which rules are what
keys_img_location_dict = {'left': 4, 'down': 2, 'right': 3, 'up': 1}
locations = ['up', 'down', 'right', 'left', 'intercept']

kf = KFold(n_splits=5)

# We've gotten the folds
for i, (train_index, test_index) in enumerate(kf.split(Xdata, y=Ydata)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    Xtraindata = Xdata[train_index, :, :].reshape((len(features), -1, len(locations)))
    Xtestdata = Xdata[test_index, :, :].reshape((len(features), -1, len(locations)))
    Ytraindata = np.swapaxes(Ydata[train_index, :], 0, 1)
    Ytestdata = np.swapaxes(Ydata[test_index, :], 0, 1)

    numInput = len(locations)
    numTrialTrain = Xtraindata.shape[1]*Xtraindata.shape[0]
    numTrialTest = Xtestdata.shape[1]*Xtestdata.shape[0]
    Xtraindata = list(Xtraindata)
    Xtestdata = list(Xtestdata)
    Ytraindata = list(Ytraindata)
    Ytestdata = list(Ytestdata)
    glmhmmOne = ssm.HMM(1, 1, numInput, observations="input_driven_obs", observation_kwargs=dict(C=2),
                        transitions='inputdriven')
    fit_ll_One = glmhmmOne.fit(Ytraindata, inputs=Xtraindata, method='em', num_iters=2,tolerance=10**-4)

    glmhmm = ssm.HMM(3, 1, numInput, observations="input_driven_obs", observation_kwargs=dict(C=2),
                        transitions='inputdriven')

    glmhmm = initializeObservation(glmhmm, glmhmmOne, observation_noise)
    glmhmm = initializeTransition(glmhmm, glmhmmOne, diagonal_p, wzero)
    fit_ll = glmhmm.fit(Ytraindata, inputs=Xtraindata, method='adam', num_iters=10000)

    # Get performance
    train_ll = glmhmm.log_likelihood(Ytraindata, inputs=Xtraindata) / numTrialTrain
    test_ll = glmhmm.log_likelihood(Ytestdata, inputs=Xtestdata) / numTrialTest

    print(train_ll)
    print(test_ll)
    save_name = (f'glmtype{glmType}_lag{glmLag}_intercept{intercept}_kfold{i}_numstates{num_states}_observationnoise'
                 f'{observation_noise}_diagonalp{diagonal_p}_wzero{wzero}.pickle')
    with open(save_directory / save_name, 'wb') as f:
        pickle.dump([glmhmm, fit_ll, train_ll, test_ll, numTrialTrain, numTrialTest], f)


# print(train)
print('bruh')