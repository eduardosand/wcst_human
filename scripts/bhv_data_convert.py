from behavior_analysis import process_wcst_behavior
from pathlib import Path
import os
import numpy as np
import pickle


def bhv_convert(subject, session, intercept, save_directory):
    """
    Converts the behavioral data to a format suitable for GLMHMM modelling. Gets the choice and outcome for each feature
    from previous trial as historydata, and the current choice as choice data. Note this assumes lag of 1, for lag of 2,
    this function will need to be adjusted.

    :param subject: (string) subject name
    :param session: (string) session name
    :param intercept: (int) probably 1 or 0
    :return: history_data: (array) n_trials X n_features X n_history_features - tells you the outcome for each trial
    and feature.
    :return: choice_data: (array) n_trials X n_features - tells us on a given trial whether a feature was chosen
    :return: history_possibilities: label each part(technically feature, but not the same feature as in the task) of the
    history data, specifically labels each index in the n_history_features to tell you what it means
    :return object_choice_data: (array) n_cards X n_trials - tells you which card was chosen
    """

    data_directory = Path(f"{os.pardir}/data/{subject}/{session}/")
    beh_directory = data_directory / "behavior"

    # Will Need to be changed
    beh_data, rule_shifts_ind, _ = process_wcst_behavior(beh_directory / f"{subject}_wcst6_2020_Aug_20_1802.csv")
    beh_data.set_index(['trial'], inplace=True)

    # r = beh_data['correct']
    # rule_dict = {'S': 'Problem', 'T': 'Shape', 'C': 'Shape', 'Q': 'Shape', 'B': 'Color', 'Y': 'Color', 'G': 'Color',
    #                      'M': 'Color', 'L': 'Texture', 'P': 'Texture', 'W': 'Texture', 'R': 'Texture'}
    features = ['S0', 'T', 'C', 'Q', 'B', 'Y', 'G', 'M', 'L', 'P', 'S2', 'R']
    feature_dict = dict(zip(features, np.linspace(0,12)))
    # then create an array but with keys so we can keep track of which rules are what
    keys_img_location_dict = {'left': 4, 'down': 2, 'right': 3, 'up': 1}
    locations = ['up', 'down', 'right', 'left']
    # now we imagine the following scenario
    # history with one lag can be one of four options
    # NC-, NC+, C-, C+
    # corresponding to 2D vectors r_t-1, c_t-1
    # (0,0), (1,0), (0,1), (1,1)
    lag = 1
    history_choice_dict = {(0, 0): 'NC-', (1, 0): 'NC+', (0, 1): 'C-', (1, 1): 'C+'}
    history_possibilities = ['intercept', 'NC-', 'NC+', 'C-', 'C+']
    # One of these for each feature
    history_data = np.zeros((len(beh_data.index.values), len(features), len(history_possibilities)))
    choice_data = np.zeros((len(beh_data.index.values), len(features), 1), dtype=np.int8) # feature level
    object_choice_data = np.zeros((len(locations), len(beh_data.index.values)), dtype=np.int8)
    stimulus_data = np.zeros((len(features), len(locations), len(beh_data.index.values)))
    for ind, row in enumerate(beh_data.index.values):
        # rule = beh_data.loc[row, 'rule']
        # connect this to what 'card' was correct
        chosen_card = beh_data.loc[row, 'chosen']
        # object choice?
        object_choice = [i for i in range(4) if beh_data.loc[row, f'bmp_table_{i+1}'] == chosen_card + ".bmp"][0]
        object_choice_data[object_choice, ind] = 1

        reward = beh_data.loc[row, 'correct']
        chosen_ind = [features.index(s) if s != 'S' else features.index(f"S{chosen_card.index(s)}") for s in chosen_card]
        # print(chosen_ind)
        # do this across trials
        choice_data[ind, chosen_ind] = 1.

        # collect stimuli for this trial on an object level
        object_stimuli = [beh_data.loc[row, f'bmp_table_{i+1}'][:-4] for i in range(4)]
        # for each feature and trial, we need to see if it was chosen and rewarded
        for feature_ind, feature in enumerate(features):
            # to update stimulus data first find the card associated with that feature
            if feature in ['S0', 'S2']:
                # looking for S in two different spots
                card_num = [i for i in range(len(object_stimuli)) if feature[0] in object_stimuli[i][int(feature[1])]][0]
                stimulus_data[feature_ind, card_num, ind] = 1.
            else:
                # looking for feature in card
                card_num = [i for i in range(len(object_stimuli)) if feature in object_stimuli[i]][0]
                stimulus_data[feature_ind, card_num, ind] = 1.

            # look for feature in chosen card, but problem if feature is S0 or S2
            if feature.startswith('S'):
                if 'S' in chosen_card:
                    s_ind = chosen_card.index('S')
                    if feature == f'S{s_ind}':
                        chosen = 1
                    else:
                        chosen = 0
                else:
                    chosen = 0
            elif feature in chosen_card:
                chosen = 1
            else:
                chosen = 0

            feature_history = history_choice_dict[(reward, chosen)]
            history_data[ind, feature_ind, history_possibilities.index(feature_history)] = 1.

    history_data[:, :, 0] = intercept
    # Final step, drop the first trial because there's no history of choice from the last trial
    choice_data = choice_data[1:, :, :]
    object_choice_data = object_choice_data[:, 1:]
    stimulus_data = stimulus_data[:, :, 1:]
    # and then drop the last trial for X because it won't predict the choice on the next trial (the next trial doesn't exist)
    history_data = history_data[:-1, :, :]

    # save pickle file so this code doesn't need to run
    save_name = (f'{subject}_{session}_glm1_lag{lag}_int{intercept}.pickle')
    with open(save_directory / save_name, 'wb') as f:
        pickle.dump([history_data, choice_data, history_possibilities, object_choice_data, locations, stimulus_data], f)
    return history_data, choice_data, history_possibilities, object_choice_data, locations, stimulus_data
