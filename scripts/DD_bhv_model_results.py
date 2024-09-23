# Goal of this analysis is just to plot all the log likelihood averaging across folds
from pathlib import Path
import os
import pickle
import numpy as np

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
data = []
for intercept in intercepts:
    for num_states in num_states_poss:
        for i in range(5):
            train_ll_avgs = []
            test_ll_avgs = []
            # best_train_save_name = (f'best_train_glmtype{glmType}_lag{glmLag}_intercept{intercept}_kfold{i}'
            #                         f'_numstates{num_states}_observationnoise{observation_noise}_diagonalp'
            #                         f'{diagonal_p}_wzero{wzero}.pickle')
            # with open(save_directory / best_train_save_name, 'wb') as f:
            #     _, best_train_fit_ll, best_train_train_ll, best_train_test_ll, best_train_numTrialTrain, best_train_numTrialTest, train_ll_avg, test_ll_avg = pickle.load(f)
            best_test_save_name = (f'best_test_glmtype{glmType}_lag{glmLag}_intercept{intercept}_kfold{i}'
                                   f'_numstates{num_states}_observationnoise{observation_noise}_diagonalp'
                                   f'{diagonal_p}_wzero{wzero}.pickle')
            with open(save_directory / best_test_save_name, 'wb') as f:
                _, best_test_fit_ll, best_test_train_ll, best_test_test_ll, best_test_numTrialTrain, best_test_numTrialTest, train_ll_avg, test_ll_avg = pickle.load(f)
                train_ll_avgs.append(train_ll_avg)
                test_ll_avgs.append(test_ll_avg)

        # Average across folds
        avg_train_ll = np.mean(train_ll_avgs)
        avg_test_ll = np.mean(test_ll_avgs)

        # Append the data for plotting
        data.append({
                    'intercept': intercept,
                    'num_states': num_states,
                    'log_likelihood': avg_train_ll,
                    'type': 'Train'
                })
        data.append({
                    'intercept': intercept,
                    'num_states': num_states,
                    'log_likelihood': avg_test_ll,
                    'type': 'Test'
                })

# Convert the list of dictionaries into a DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))
sns.violinplot(x='num_states', y='log_likelihood', hue='type', data=df, split=True)
plt.title('Train and Test Log-Likelihood Averages by Number of States and Intercepts')
plt.xlabel('Number of States')
plt.ylabel('Log-Likelihood')
plt.legend(title='Type')
plt.show()