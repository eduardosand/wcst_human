# Goal of this analysis is just to plot all the log likelihood averaging across folds
from pathlib import Path
import os
import pickle
import numpy as np

intercepts = [1]
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
        train_ll_avgs = []
        test_ll_avgs = []
        for i in range(5):
            # best_train_save_name = (f'best_train_glmtype{glmType}_lag{glmLag}_intercept{intercept}_kfold{i}'
            #                         f'_numstates{num_states}_observationnoise{observation_noise}_diagonalp'
            #                         f'{diagonal_p}_wzero{wzero}.pickle')
            # with open(save_directory / best_train_save_name, 'wb') as f:
            #     _, best_train_fit_ll, best_train_train_ll, best_train_test_ll, best_train_numTrialTrain, best_train_numTrialTest, train_ll_avg, test_ll_avg = pickle.load(f)
            best_test_save_name = (f'best_test_glmtype{glmType}_lag{glmLag}_intercept{intercept}_kfold{i}'
                                   f'_numstates{num_states}_observationnoise{observation_noise}_diagonalp'
                                   f'{diagonal_p}_wzero{wzero}').replace('.','p')
            best_test_save_name = best_test_save_name + '.pickle'
            with open(save_directory / best_test_save_name, 'rb') as f:
                print('yay')
                [_, best_test_fit_ll, best_test_train_ll, best_test_test_ll, best_test_numTrialTrain, best_test_numTrialTest, train_ll_avg, test_ll_avg] = pickle.load(f)
                train_ll_avgs.append(train_ll_avg)
                test_ll_avgs.append(test_ll_avg)

        # Average across folds
        # avg_train_ll = np.mean(train_ll_avgs)
        # avg_test_ll = np.mean(test_ll_avgs)

        # Append the data for plotting
                data.append({
                    'intercept': intercept,
                    'num_states': num_states,
                    'log_likelihood': train_ll_avg,
                    'type': 'Train'
                })
                data.append({
                    'intercept': intercept,
                    'num_states': num_states,
                    'log_likelihood': test_ll_avg,
                    'type': 'Test'
                })

# Convert the list of dictionaries into a DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.DataFrame(data)

# Prepare the data for plotting
num_states_poss = sorted(df['num_states'].unique())
train_data = [df[(df['num_states'] == num_states) & (df['type'] == 'Train')]['log_likelihood'].values
              for num_states in num_states_poss]
test_data = [df[(df['num_states'] == num_states) & (df['type'] == 'Test')]['log_likelihood'].values
              for num_states in num_states_poss]

# Create violin plots
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Train data as violins
parts_train = ax.violinplot(train_data, positions=np.array(num_states_poss) - 0.15, widths=0.3, showmeans=True, showmedians=True)
for pc in parts_train['bodies']:
    pc.set_facecolor('blue')
    pc.set_edgecolor('black')
    pc.set_alpha(0.6)

# Plot Test data as violins
parts_test = ax.violinplot(test_data, positions=np.array(num_states_poss) + 0.15, widths=0.3, showmeans=True,
                           showmedians=True)
for pc in parts_test['bodies']:
    pc.set_facecolor('orange')
    pc.set_edgecolor('black')
    pc.set_alpha(0.6)

# Manually add legend
train_patch = plt.Line2D([0], [0], color='blue', lw=4, label='Train')
test_patch = plt.Line2D([0], [0], color='orange', lw=4, label='Test')

# Customize the plot
ax.set_title('Train and Test Log-Likelihood Averages by Number of States and Intercepts')
ax.set_xlabel('Number of States')
ax.set_ylabel('Log-Likelihood')
ax.set_xticks(num_states_poss)
ax.legend(['Train', 'Test'], loc='upper right')

# Show plot
plt.show()