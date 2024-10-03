# Goal of this analysis is just to plot all the log likelihood averaging across folds
from pathlib import Path
import os
import pickle
import numpy as np

intercepts = [0,1]
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
                                   f'{diagonal_p}_wzero{wzero}')
            best_test_save_name = best_test_save_name + '.pkl'
            with open(save_directory / best_test_save_name, 'rb') as f:
                [_, best_test_fit_ll, best_test_train_ll, best_test_test_ll, best_test_numTrialTrain, best_test_numTrialTest, train_ll_avg, test_ll_avg, _] = pickle.load(f)
                train_ll_avgs.append(train_ll_avg)
                test_ll_avgs.append(test_ll_avg)

        # Average across folds
        # avg_train_ll = np.mean(train_ll_avgs)
        # avg_test_ll = np.mean(test_ll_avgs)
                num_parameters = 5*num_states+5*num_states**2 + (1+num_states+num_states**2)*intercept
                # Append the data for plotting
                data.append({
                    'intercept': intercept,
                    'num_states': num_states,
                    'log-likelihood': train_ll_avg,
                    'type': 'Train',
                    'num_parameters': num_parameters,
                    'BIC': num_parameters*np.log(199*12)-2*train_ll_avg,
                    'AIC': 2*num_parameters - 2 * train_ll_avg
                })
                data.append({
                    'intercept': intercept,
                    'num_states': num_states,
                    'log-likelihood': test_ll_avg,
                    'type': 'Test',
                    'num_parameters': num_parameters,
                    'BIC': num_parameters*np.log(199*12)-2*test_ll_avg,
                    'AIC': 2*num_parameters - 2 * test_ll_avg
                })

# Convert the list of dictionaries into a DataFrame
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(data)


def model_comparison(df, selection, key_var='num_states', key_var_constant='intercept', constant_value=0):
    # Prepare the data for plotting
    num_states_poss = sorted(df[key_var].unique())
    train_data = [df[(df[key_var] == num_states) & (df['type'] == 'Train') & (df[key_var_constant] == constant_value)][selection].values
                  for num_states in num_states_poss]
    test_data = [df[(df[key_var] == num_states) & (df['type'] == 'Test') & (df[key_var_constant] == constant_value)][selection].values
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
    ax.legend(handles=[train_patch, test_patch], loc='lower right', fontsize=20)


    # Customize the plot
    if key_var == 'num_states':
        first_string = 'Number of States'
    else:
        first_string = 'Intercept'

    if key_var_constant == 'num_states':
        second_string = 'Number of States'
    else:
        second_string = 'Intercept'
    ax.set_title(f'Train and Test {selection} Averages by {first_string} \n holding {second_string} at {constant_value}', fontsize=28)
    ax.set_xlabel(f'{first_string}', fontsize=20)
    ax.set_ylabel(f'{selection}', fontsize=20)
    ax.set_xticks(num_states_poss)

    save_directory = Path(f"{os.pardir}/data/{subject}/{session}/model")
    save_name = f'bhv_model_{selection}_{key_var}_constant{key_var_constant}_{constant_value}.svg'
    plt.tight_layout()
    # Show plot
    plt.savefig(save_directory / save_name)
    plt.show()




# Less step look at AIC and BIC
key_word = 'num_states'
key_word_constant = 'intercept'
for i in range(2):
    constant_value = i
    model_comparison(df, 'log-likelihood', key_word, key_word_constant, constant_value)
    model_comparison(df, 'BIC', key_word, key_word_constant, constant_value)
    model_comparison(df, 'AIC', key_word, key_word_constant, constant_value)
key_word = 'intercept'
key_word_constant = 'num_states'
for i in range(1, 5):
    constant_value = i
    model_comparison(df, 'log-likelihood', key_word, key_word_constant, constant_value)
    model_comparison(df, 'BIC', key_word, key_word_constant, constant_value)
    model_comparison(df, 'AIC', key_word, key_word_constant, constant_value)
