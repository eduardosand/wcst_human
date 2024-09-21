from behavior_analysis import process_wcst_behavior
import os
from pathlib import Path
import numpy as np
import ssm
from bhv_data_convert import bhv_convert
import pickle
from sklearn.model_selection import KFold
"""
Some of this code adapted from 
A Comparison of Rapid Rule-Learning Strategies in Humans and Monkeys
Vishwa Goudar, Jeong-Woo Kim, Yue Liu, Adam J. O. Dede, Michael J. Jutras, Ivan Skelin, Michael Ruvalcaba, William Chang, Bhargavi Ram, Adrienne L. Fairhall, Jack J. Lin, Robert T. Knight, Elizabeth A. Buffalo, Xiao-Jing Wang
Journal of Neuroscience 10 July 2024, 44 (28) e0231232024; DOI: 10.1523/JNEUROSCI.0231-23.2024
"""


def initializeObservation(glmhmm,glmhmmOne,sigma, rng_generator):
    """
    function to initialize observation weights of glmhmm
    :param glmhmm:
    :param glmhmmOne:
    :param sigma:
    :return:
    """

    for k in range(glmhmm.K):
        obs_shape = glmhmmOne.observations.params[0].shape
        glmhmm.observations.params[k] = glmhmmOne.observations.params[0] * (1 + sigma *
                                                                            rng_generator.random(obs_shape))

    return glmhmm


def initializeTransition(glmhmm,glmhmmOne,diagonalP,wZero, rng_generator):
    """
    function to initialize transition weights of glmhmm
    :param glmhmm:
    :param glmhmmOne:
    :param diagonalP:
    :param wZero:
    :return:
    """

    # initialize log_Ps variable
    Ps = diagonalP * np.eye(glmhmm.K) + .05 * rng_generator.random((glmhmm.K, glmhmm.K))
    Ps /= Ps.sum(axis=1, keepdims=True)
    glmhmm.transitions.log_Ps = np.log(Ps)

    # initialize Ws variable
    if wZero == 1:
        glmhmm.transitions.Ws = np.zeros(glmhmm.transitions.Ws.shape)
    elif wZero == 2:
        for k1 in range(glmhmm.K):
            for k2 in range(glmhmm.K):
                glmhmm.transitions.Ws[k1, k2, :] = glmhmmOne.observations.params[0]

    return glmhmm


def main():
    # Goals of the analysis:
    # Adjudicate between fit HMMs with different intercepts and number of states
    intercepts = [0, 1]
    num_states_poss = [1, 2, 3, 4]
    seeds = [1234567890, 2345678901, 3456789012, 4567890123, 5678901234, 6789012345, 7890123456, 8901234567,
             9012345678, 12345678900]
    subject = 'IR95'
    session = 'sess-3'
    save_directory = Path(f"{os.pardir}/data/{subject}/{session}/model")

    glmType = 1  # Bernoulli
    glmLag = 1 # Is it Markov or more
    observation_noise = 0
    diagonal_p = 0.9
    wzero = 1

    for intercept in intercepts:

        # convert raw behavior data into array encoded info
        # history vector is reward and choice encoded as one hot vector with an intercept
        # history labels make it clear what option pertains to the label
        # choice is a single number
        history_data, choice_data, history_labels, _, _, _ = bhv_convert(subject, session, intercept, save_directory)

        Xdata = history_data
        Ydata = choice_data

        for num_states in num_states_poss:
            # then map the rule to these keys to replace some things with 1
            rule_dict = {'S': 'Problem', 'T': 'Shape', 'C': 'Shape', 'Q': 'Shape', 'B': 'Color', 'Y': 'Color', 'G': 'Color',
                                 'M': 'Color', 'L': 'Texture', 'P': 'Texture', 'W': 'Texture', 'R': 'Texture'}
            features = ['S0', 'T', 'C', 'Q', 'B', 'Y', 'G', 'M', 'L', 'P', 'S2', 'R']
            feature_dict = dict(zip(features, np.linspace(0, 12)))

            kf = KFold(n_splits=5)

            # We've gotten the folds
            for i, (train_index, test_index) in enumerate(kf.split(Xdata, y=Ydata)):
                print(f"Fold {i}:")
                print(f"  Train: index={train_index}")
                print(f"  Test:  index={test_index}")
                Xtraindata = Xdata[train_index, :, :].reshape((len(features), -1, len(history_labels)))
                Xtestdata = Xdata[test_index, :, :].reshape((len(features), -1, len(history_labels)))
                Ytraindata = np.swapaxes(Ydata[train_index, :], 0, 1)
                Ytestdata = np.swapaxes(Ydata[test_index, :], 0, 1)

                numInput = len(history_labels)
                numTrialTrain = Xtraindata.shape[1]*Xtraindata.shape[0]
                numTrialTest = Xtestdata.shape[1]*Xtestdata.shape[0]
                Xtraindata = list(Xtraindata)
                Xtestdata = list(Xtestdata)
                Ytraindata = list(Ytraindata)
                Ytestdata = list(Ytestdata)
                train_ll_benchmark = -100000000
                test_ll_benchmark = -1000000000
                train_ll_sum = 0
                test_ll_sum = 0
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    glmhmmOne = ssm.HMM(1, 1, numInput, observations="input_driven_obs", observation_kwargs=dict(C=2),
                                        transitions='inputdriven')
                    fit_ll_One = glmhmmOne.fit(Ytraindata, inputs=Xtraindata, method='em', num_iters=2, tolerance=10**-4)

                    glmhmm = ssm.HMM(num_states, 1, numInput, observations="input_driven_obs", observation_kwargs=dict(C=2),
                                        transitions='inputdriven')

                    glmhmm = initializeObservation(glmhmm, glmhmmOne, observation_noise, rng)
                    glmhmm = initializeTransition(glmhmm, glmhmmOne, diagonal_p, wzero, rng)
                    fit_ll = glmhmm.fit(Ytraindata, inputs=Xtraindata, method='adam', num_iters=10000)

                    # Get performance
                    train_ll = glmhmm.log_likelihood(Ytraindata, inputs=Xtraindata) / numTrialTrain
                    test_ll = glmhmm.log_likelihood(Ytestdata, inputs=Xtestdata) / numTrialTest

                    # we'd like to save intermittently the best model over all the initializations
                    if train_ll > train_ll_benchmark:
                        best_train_glmhmm = glmhmm
                        best_train_fit_ll = fit_ll
                        train_ll_benchmark = train_ll
                        best_train_train_ll = train_ll
                        best_train_test_ll = test_ll
                        best_train_numTrialTrain = numTrialTrain
                        best_train_numTrialTest = numTrialTest
                    if test_ll > test_ll_benchmark:
                        best_test_glmhmm = glmhmm
                        best_test_fit_ll = fit_ll
                        test_ll_benchmark = test_ll
                        best_test_train_ll = train_ll
                        best_test_test_ll = test_ll
                        best_test_numTrialTrain = numTrialTrain
                        best_test_numTrialTest = numTrialTest

                    # we also want the average among these, technically an average of averages isn't clean
                    # but this is the way it looks they did it in the paper
                    print(train_ll)
                    print(test_ll)
                    train_ll_sum += train_ll
                    test_ll_sum += test_ll
                train_ll_avg = train_ll_sum / len(seeds)
                test_ll_avg = test_ll_sum / len(seeds)
                best_train_save_name = (f'best_train_glmtype{glmType}_lag{glmLag}_intercept{intercept}_kfold{i}'
                                        f'_numstates{num_states}_observationnoise{observation_noise}_diagonalp'
                                        f'{diagonal_p}_wzero{wzero}.pickle')
                with open(save_directory / best_train_save_name, 'wb') as f:
                        pickle.dump([best_train_glmhmm, best_train_fit_ll, best_train_train_ll, best_train_test_ll,
                                     best_train_numTrialTrain, best_train_numTrialTest, train_ll_avg, test_ll_avg], f)
                best_test_save_name = (f'best_test_glmtype{glmType}_lag{glmLag}_intercept{intercept}_kfold{i}'
                                       f'_numstates{num_states}_observationnoise{observation_noise}_diagonalp'
                                       f'{diagonal_p}_wzero{wzero}.pickle')
                with open(save_directory / best_test_save_name, 'wb') as f:
                        pickle.dump([best_test_glmhmm, best_test_fit_ll, best_test_train_ll, best_test_test_ll,
                                     best_test_numTrialTrain, best_test_numTrialTest, train_ll_avg, test_ll_avg], f)



if __name__ == "__main__":
    main()
