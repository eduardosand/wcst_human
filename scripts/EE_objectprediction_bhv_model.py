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
from pathlib import Path
import os
from bhv_data_convert import bhv_convert


mpl.rcParams['agg.path.chunksize'] = 10000


def getDataToSuperBlocks(historyData, choiceData, intercept, mode):
    """
    This is gonna transform the behavioral data into a format suitable for analysis and extension models.
    For Vishwa's paper, he uses SuperBlocks, but here I have one block, so a mode parameter is added to account for
    this discrepancy.
    :param historyData: (ndarray) n_trials x n_features x n_labels(labels are intercept and which history(NC-,C+, etc),
    pertains the previous trial
    :param choiceData: (ndarray) n_trials x n_features x 1: whether each feature was part of the chosen object,
    for each trial
    :param intercept: if intercept is 1 or 0
    :param mode:
    :return:
    """
    if mode == 'Vishwa':
        numBlocks = 1
    else:
        choiceData = np.expand_dims(choiceData, axis=0)
        historyData = np.expand_dims(historyData, axis=0)
        # numBlocks = len(historyData)


    historySuperBlocks = []
    choiceSuperBlocks = []

    # Loop through super blocks
    # (i.e. series of trials without breaks - may span multiple rules)
    for blk in range(numBlocks):
        T = historyData[blk][0].shape[0]
        H = np.zeros([12, T])
        C = np.zeros([12, T])
        # Loop through features and transform to arrays
        for f in range(12):
            C[f, :] = np.squeeze(choiceData[blk][f].astype('int'))
            H[f, :] = np.argmax(historyData[blk][f][:, intercept:], axis=1).astype('int')
        historySuperBlocks.append(H)
        choiceSuperBlocks.append(C)

    return historySuperBlocks, choiceSuperBlocks


def get_choice_likelihood(glmhmm = None,
                        intercept = 1,
                        states = None,
                        history = None,
                        choice = None,
                        lag = 1,
                        K=1,
                        empirical = False):
    """
    Get the choice likelihood given a model, a certain dataset.
    :param glmhmm:
    :param intercept:
    :param states:
    :param history:
    :param choice:
    :param lag:
    :param K:
    :param empirical:
    :return:
    """

    if empirical: # from empirical measurement

        pc_HS = np.zeros([2, 4 ** lag, K])

        numBlocks = len(states)

        # Loop through super blocks
        # (i.e. series of trials without breaks - may span multiple rules)
        for blk in range(numBlocks):
            # Loop through states
            for s in range(K):
                for h in range(pc_HS.shape[1]):
                    sBlk = states[blk] == s
                    hBlk = history[blk] == h
                    hsBlk = np.multiply(hBlk, sBlk)
                    pc_HS[1, h, s] += np.count_nonzero(hsBlk)
                    pc_HS[0, h, s] += np.count_nonzero(np.multiply(hsBlk, choice[blk]))

        pc_HS = np.divide(pc_HS[0, :, :], pc_HS[1, :, :])
        pc_HS[np.isnan(pc_HS)] = 0

    else: #  from (fit) model parameters
        Wk = glmhmm.observations.Wk
        # pc_HS_mdl = np.zeros([4**lag,K])

        if intercept == 1:
            x = Wk[:, 0, :1] + Wk[:, 0, 1:]
        else:
            x = Wk[:, 0, :]
        pc_HS = 1 / (1 + np.exp(x.T))

    return pc_HS


def object_prediction_set_up(subject, session, intercept, kfoldnum, num_states, pos=False):
    """
    Code exists to turn basic behavioral data and instance of glmhmm into necessary information useful
    for object prediction extension model.
    :return:
    """
    # first thing we need is all our necessary components from behavioral data conversion
    glmType = 1  # Bernoulli
    glmLag = 1
    # num_states = 2
    observation_noise = 0
    diagonal_p = 0.9
    wzero = 1
    lag = 1
    save_directory = Path(f"{os.pardir}/data/{subject}/{session}/model")
    save_name = (
        f'glmtype{glmType}_lag{glmLag}_intercept{intercept}_kfold{kfoldnum}_numstates{num_states}_observationnoise'
        f'{observation_noise}_diagonalp{diagonal_p}_wzero{wzero}.pickle')

    with open(save_directory / save_name, 'rb') as f:
        try:
            [glmhmm, fit_ll, train_ll, test_ll, trCnt, teCnt] = pickle.load(f)
        except:
            [glmhmm, fit_ll, train_ll, test_ll] = pickle.load(f)

    save_name_bhv = Path(f"{subject}_{session}_glm1_lag{lag}_int{intercept}.pickle")
    with open(save_directory / save_name_bhv, 'rb') as f:
        history_data, choice_data, history_labels, object_choice, locations, stimulus_data = pickle.load(f)
    # history_data, choice_data = bhv_convert(subject, session, intercept)
    # choice_data (array) - n_trials X n_features X n_blocks

    # we need stimulus super block in order to construct our feature choice likelihoods
    num_trials = choice_data.shape[0]



    viterbi_block = np.array([glmhmm.most_likely_states(choice_data[:, f, :].astype('int'),
                                                        input=history_data[:, f, :]) for f in range(12)])

    # posterior probabilities for each thing given each state
    posterior_block = np.array([glmhmm.expected_states(choice_data[:, f, :].astype('int'),
                                                       input=history_data[:, f, :])[0]
                                for f in range(12)])
    # model choice likelihood
    pc_HS_mdl = get_choice_likelihood(glmhmm=glmhmm, intercept=intercept)



    feature_choice_likelihood = np.zeros([4,3,num_trials])
    feature_choice_likelihood_pos = np.zeros([4,3, num_trials])

    for t in range(num_trials):
        for i in range(4):
            idxF = np.where(stimulus_data[:, i, t] == 1)[0]
            for j in range(len(idxF)):
                f = idxF[j]
                H = np.argmax(history_data[t,f, intercept:]).astype('int')
                S = int(viterbi_block[f,t])
                # H = int(superBlocksData['history'][blk][f, t])
                # S = int(superBlocksData['viterbi'][blk][f, t])
                print(H)
                print(S)
                feature_choice_likelihood[i, j, t] = pc_HS_mdl[H, S]
                feature_choice_likelihood_pos[i, j, t] = np.sum(pc_HS_mdl[H, :] * posterior_block[f, t, :])

    # next piece taken from Vishwa
    if pos:  # version based on posterior
        feature_choice_likelihood = feature_choice_likelihood_pos

    # Randomize position of objects in input structure
    feature_choice_likelihood = np.swapaxes(feature_choice_likelihood, 1, 2)
    for i in range(feature_choice_likelihood.shape[1]):
        pInds = np.random.permutation(4)
        object_choice[:, i] = object_choice[pInds, i]
        tmp = np.copy(feature_choice_likelihood[:, i, :])
        for j in range(4):
            feature_choice_likelihood[j, i, :] = tmp[pInds[j], :]

    # trials x num features (ordered in consecutive sets of 3 per object)
    feature_choice_likelihood = np.hstack(feature_choice_likelihood)
    print(feature_choice_likelihood.shape)
    # trials x num objects
    object_choice = np.swapaxes(object_choice, 0, 1)

    # trials
    object_choice_index = np.nonzero(object_choice)[1]
    return feature_choice_likelihood, object_choice_index, object_choice


class NetSmall(nn.Module):

    def __init__(self, msk):
        super(NetSmall, self).__init__()
        self.fc1 = nn.Linear(12, 4)
        if msk:
            conn = np.zeros((12,4), dtype = np.int32)
            conn[:3,0] = 1
            conn[3:6,1] = 1
            conn[6:9,2] = 1
            conn[9:,3] = 1
            prune.custom_from_mask(self.fc1, name='weight', mask=torch.tensor(conn.T))
        self.mx = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        y = self.mx(x)
        return x, y

# test Net
net = NetSmall(msk=True)
print(net)
params = list(net.parameters())
for p in params:
    print(p.size())


def setupData(species, subj, pos=False):
    # Load data
    featureChoiceLikelihood, featureChoiceLikelihood_pos, objectChoice, rules = pickle.load(
        open('rawData/forObjectPred/' + species + '/' + subj + '_objPredFromFeatPred.pickle', "rb"))

    if pos:  # version based on posterior
        featureChoiceLikelihood = featureChoiceLikelihood_pos

    # Randomize position of objects in input structure
    featureChoiceLikelihood = np.swapaxes(featureChoiceLikelihood, 1, 2)
    for i in range(featureChoiceLikelihood.shape[1]):
        pInds = np.random.permutation(4)
        objectChoice[:, i] = objectChoice[pInds, i]
        tmp = np.copy(featureChoiceLikelihood[:, i, :])
        for j in range(4):
            featureChoiceLikelihood[j, i, :] = tmp[pInds[j], :]

    # trials x num features (ordered in consecutive sets of 3 per object)
    featureChoiceLikelihood = np.hstack(featureChoiceLikelihood)
    print(featureChoiceLikelihood.shape)
    # trials x num objects
    objectChoice = np.swapaxes(objectChoice, 0, 1)

    # Also get index of chosen object
    objectChoice0 = objectChoice
    # trials
    objectChoice = np.nonzero(objectChoice)[1]

    return featureChoiceLikelihood, objectChoice, objectChoice0


def trainNet(seed, featureChoiceLikelihood, objectChoice, objectChoice0, testInds):
    # create network
    torch.manual_seed(seed)
    net = NetSmall(msk=True)

    learningRate = 0.001
    batchSize = 1000
    np.random.seed(seed)
    np.set_printoptions(threshold=sys.maxsize)

    # create  optimizer
    optimizer = optim.Adam(net.parameters(), lr=learningRate)
    criterion = nn.CrossEntropyLoss()

    # for metrics
    losses = []
    perfTrain = []
    perfTest = []

    # train-validation split
    trainInds = np.setdiff1d(np.arange(objectChoice.shape[0]), testInds)
    trainInputs = np.log(featureChoiceLikelihood[trainInds, :])
    trainTargets = objectChoice[trainInds]
    trainTargets0 = objectChoice0[trainInds, :]

    testInputs = np.log(featureChoiceLikelihood[testInds, :])
    testTargets = objectChoice[testInds]
    testTargets0 = objectChoice0[testInds, :]
    print([objectChoice.shape[0], trainInds.shape, testInds.shape])

    # Loop through training epochs
    for lp in range(100000):
        net.train()

        # create mini-batch
        inds = np.random.randint(0, high=trainTargets.shape[0], size=batchSize)
        input = torch.tensor(trainInputs[inds, :], dtype=torch.float)
        target = torch.tensor(trainTargets[inds], dtype=torch.long)

        optimizer.zero_grad()  # zero the gradient buffers
        outputRaw, output = net(input)  # forward pass
        loss = criterion(outputRaw, target)  # loss
        losses.append(loss.detach().numpy().item(0))
        output = output.detach().numpy()
        pp = output * trainTargets0[inds, :]

        loss.backward()  # backward pass
        optimizer.step()  # weight update

        net.eval()
        # Evaluate on all of the training set
        pp = []
        for indStart in range(0, trainInputs.shape[0], 10000):
            indEnd = min(indStart + 10000, trainInputs.shape[0])
            input = torch.tensor(trainInputs[indStart:indEnd, :], dtype=torch.float)
            outputRaw, output = net(input)
            output = output.detach().numpy()
            pp_sub = np.max(output * trainTargets0[indStart:indEnd, :], axis=1)
            pp.append(pp_sub)
        perfTrain.append(np.mean(np.concatenate(pp)))

        # Evaluate on the validation set
        input = torch.tensor(testInputs, dtype=torch.float)
        outputRaw, output = net(input)
        output = output.detach().numpy()
        pp = output * testTargets0
        perfTest.append(np.mean(np.max(pp, axis=1)))

        # Update live plot
        # if lp % 100 == 0:
        #     live_plot(perfTrain, perfTest)

        # Check for convergence and return
        if len(perfTrain) > 500 and (
                (np.sum(np.greater(np.array(perfTrain[-500:]), np.array(perfTest[-500:]) + 5)) == 500) or (
                np.abs(np.mean(perfTest[-500:-250]) - np.mean(perfTest[-250:])) < 0.0005) or (
                        np.mean(perfTest[-250:]) > 0.999)):
            bestTrain = perfTrain[-1]
            bestTest = perfTest[-1]

            # Get performance on entire dataset
            pp = []
            for indStart in range(0, featureChoiceLikelihood.shape[0], 10000):
                indEnd = min(indStart + 10000, featureChoiceLikelihood.shape[0])
                input = torch.tensor(np.log(featureChoiceLikelihood[indStart:indEnd, :]), dtype=torch.float)
                outputRaw, output = net(input)
                output = output.detach().numpy()
                pp_sub = np.max(output * objectChoice0[indStart:indEnd, :], axis=1)
                pp.append(pp_sub)
            allPerf = np.concatenate(pp)
            break
    return bestTrain, bestTest, allPerf

# subjs = ['sam', 'tabitha', 'chloe', 'blanche', 'b01', 'b02', 'b03', 'b04', 'b05']
# sps = ['monkey', 'monkey', 'monkey', 'monkey', 'human', 'human', 'human', 'human', 'human']
#
# for pos in [True, False]: # with and without posterior
#     bestTrains = np.zeros((9,5))
#     bestTests = np.zeros((9,5))
#     allPerfs = []
#     for sind, subj in enumerate(subjs): # Loop through subjects
#         # setup data
#         featureChoiceLikelihood, objectChoice, objectChoice0 = setupData(sps[sind], subj, pos)
#         availInds = np.arange(objectChoice.shape[0])
#
#         # Five fold cross-validation
#         for i in range(5):
#             np.random.seed(123)
#             if availInds.shape[0] > 0.3*objectChoice.shape[0]:
#                 testInds = np.random.choice(availInds, size=int(0.2*objectChoice.shape[0]), replace=False)
#             else:
#                 testInds = availInds
#             availInds = np.setdiff1d(availInds, testInds)
#
#             # Train classifier
#             btr, bte, allPerf = trainNet(i, featureChoiceLikelihood, objectChoice, objectChoice0, testInds)
#             bestTrains[sind, i] = btr
#             bestTests[sind, i] = bte
#             allPerfs.append(allPerf)
#     if pos:
#         allPerfsP = allPerfs


def main():
    subject = 'IR95'
    session = 'sess-3'
    intercept = 1
    kfoldnum = 4
    num_states = 3
    save_directory = Path(f"{os.pardir}/data/{subject}/{session}/model")
    # bhv_convert(subject, session, intercept, save_directory)

    for pos in [True, False]:  # with and without posterior
        bestTrains = np.zeros((5))
        bestTests = np.zeros((5))
        allPerfs = []
        # setup data
        feature_choice_likelihood, object_choice, object_choice_raw = object_prediction_set_up(subject, session,
                                                                                               intercept, kfoldnum,
                                                                                               num_states, pos=pos)
        # featureChoiceLikelihood, objectChoice, objectChoice0 = setupData(sps[sind], subj, pos)
        availInds = np.arange(object_choice.shape[0])

        # Five fold cross-validation
        for i in range(5):
            np.random.seed(123)
            if availInds.shape[0] > 0.3 * object_choice.shape[0]:
                testInds = np.random.choice(availInds, size=int(0.2 * object_choice.shape[0]), replace=False)
            else:
                testInds = availInds
            availInds = np.setdiff1d(availInds, testInds)

            # Train classifier
            btr, bte, allPerf = trainNet(i, feature_choice_likelihood, object_choice, object_choice_raw, testInds)
            bestTrains[i] = btr
            bestTests[i] = bte
            allPerfs.append(allPerf)
        if pos:
            allPerfsP = allPerfs

    results = {'allPerfs': allPerfs, 'allPerfsP': allPerfsP}
    with open(save_directory / 'objPredPerfs.pickle', 'wb+') as f:
        pickle.dump([results], f)

    results = {'bestTrains': bestTrains, 'bestTests': bestTests}
    with open(save_directory / 'all_objPredPerf.pickle', 'wb+') as f:
        pickle.dump([results], f)

if __name__ == "__main__":
    main()