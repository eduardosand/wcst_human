from behavior_analysis import process_wcst_behavior
from pathlib import Path
import os
import numpy as np
# Code exists to convert data into another format.
################## TO DO
# Test this code with fitting a new model and see if it works
# Need to plot the design matrix to make sure I have what I think I have


def bhv_convert(subject, session, intercept):
    """
    Converts the behavioral data to a format suitable for HMM modelling. Gets the choice and outcome for each feature
    from previous trial as historydata, and the current choice as choice data.

    :param subject:
    :param session:
    :return: history_data:
    :return: choice_data:
    """

    data_directory = Path(f"{os.pardir}/data/{subject}/{session}/")
    ph_file_path = Path("raw/Events.nev")
    beh_directory = data_directory / "behavior"
    running_avg = 5
    save_directory = Path(f"{os.pardir}/data/{subject}/{session}/model")

    # Will Need to be changed
    beh_data, rule_shifts_ind, _ = process_wcst_behavior(beh_directory / f"{subject}_wcst6_2020_Aug_20_1802.csv")
    beh_data.set_index(['trial'], inplace=True)

    r = beh_data['correct']
    # rule_dict = {'S': 'Problem', 'T': 'Shape', 'C': 'Shape', 'Q': 'Shape', 'B': 'Color', 'Y': 'Color', 'G': 'Color',
    #                      'M': 'Color', 'L': 'Texture', 'P': 'Texture', 'W': 'Texture', 'R': 'Texture'}
    features = ['S0', 'T', 'C', 'Q', 'B', 'Y', 'G', 'M', 'L', 'P', 'S2', 'R']
    feature_dict = dict(zip(features, np.linspace(0,12)))
    # then create an array but with keys so we can keep track of which rules are what
    keys_img_location_dict = {'left': 4, 'down': 2, 'right': 3, 'up': 1}
    locations = ['up', 'down', 'right', 'left', 'intercept']

    history_data = np.zeros((len(beh_data.index.values), len(features), len(locations)))
    choice_data = np.zeros((len(beh_data.index.values), len(features), 1), dtype=np.int8)
    for ind, row in enumerate(beh_data.index.values):
        # rule = beh_data.loc[row, 'rule']
        # connect this to what 'card' was correct
        chosen_card = beh_data.loc[row, 'chosen']

        chosen_ind = [features.index(s) if s != 'S' else features.index(f"S{chosen_card.index(s)}") for s in chosen_card]
        # print(chosen_ind)
        # do this across trials
        choice_data[ind, chosen_ind] = 1.
        for location in list(keys_img_location_dict.values()):
            # print(location)
            card_one = beh_data.loc[row, f'bmp_table_{location}']
            # loc_ind gives us the features for a given location
            feat_ind = np.zeros((3,), dtype=np.int8)
            if 'S' in card_one:
                # find out if there are two S or one and then what indices
                if card_one.count('S') == 2:
                    feat_ind[0] = 0
                    feat_ind[1] = 10
                else:
                    feat_ind[0] = card_one.index('S')
            card_one = card_one.replace('S', '')
            card_one = card_one.replace('.bmp', '')
            count = 3-len(card_one)
            for s in card_one:
                feat_ind[count] = features.index(s)
                count += 1
            history_data[ind, feat_ind, location-1] = 1

    history_data[:, :, 0] = intercept
    # Final step, drop the first trial because there's no history of choice from the last trial
    choice_data = choice_data[:, :, :]
    # and then drop the last trial for X because it won't predict the choice on the next trial (the next trial doesn't exist)
    history_data = history_data[:-1, :, :]
    lag = 1

    #### TO DO
    # add something in here to save the data as pickle files
    return history_data, choice_data



# for each feature, we want number_trials-1*5 array
subject = 'BERK01'
session = 'sess-1'
file_name = Path(os.pardir) / Path(f"data/{subject}/{session}/behavior/{subject}"
                                   f"_Wisconsin Card Sorting Eduardo_2024_Apr_05_0019.csv")

beh_data, _, _ = process_wcst_behavior(file_name, running_avg=5)

# Big changes between the last experiment and the current one is the number of features.
# Here we have 10 features (9 plus an empty)
# So for each feature, we should first extract the 5 row vector for each trial (essentially, what location the feature
# is in plus the intercept

print(beh_data)
##### TO DO edit this code to work with lags
