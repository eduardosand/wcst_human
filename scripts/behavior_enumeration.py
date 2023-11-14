# This script is really about taking our pre-existing behavioral analysis and enumerating conditions better


# First import in the other necessary functions
from behavior_analysis import process_wcst_behavior
import os
import numpy as np
from pathlib import Path


def feature_count(feature_list, feature_list_2=[], double=False):
    """
    Counts the number of each distinct element in list. If double boolean is used, enumerate by tuples
    :param feature_list: list of strings, int
    :param feature_list_2: optional, list of strings, int
    :param double: optional, bool
    :return: counts_dict: dict.
    """
    features = set(list(feature_list))
    counts_dict = {}
    if not double:
        feature_split = [(i, list(feature_list).count(i)) for i in features]
        counts_dict.update(feature_split)
    else:
        features_2 = set(list(feature_list_2))
        for j in features_2:
            feature_list_pruned = np.array(feature_list)[np.where(np.array(feature_list_2) == j)[0]]
            feature_split = [(f"{i}_{j}", list(feature_list_pruned).count(i)) for i in features]
            counts_dict.update(feature_split)
    return counts_dict


def enumerate_cond(beh_data):
    """
    This function serves to provide us with dictionaries that enumerate the number of trials per conditions for
    rule, feedback, rule dimension, key press, number of problems, ruleXfeedback, rule dimensionXfeedback
    for a single session
    :param beh_data:
    :return:
    """
    # next step, enumerate incorrect / correct numbers
    feedback_dict = feature_count(beh_data['correct'])
    rule_list = beh_data['rule']
    # enumerate rules / rule dimensions
    rule_dict = feature_count(rule_list)

    rule_dim_dict = feature_count(beh_data['rule dimension'])

    key_press_dict = feature_count(beh_data['key_resp_2_keys'])

    num_problems_dict = feature_count(beh_data['rule_shift_bool'])

    # preservative errors
    # defined as the continuation of a choice, long past the rule has changed.
    beh_data['preservative_error'] = 0.

    # To find the evidence of a preservative error, we'll need to find the rule_shift
    rule_shifts_ind = beh_data[beh_data.rule_shift_bool == True].index
    # rule shift corresponds to the trial after, so rule in rule_shifts_ind is the 'old' rule
    # Next check for incorrect following the rule shift
    for curr_rule in range(len(rule_shifts_ind)):
        old_rule = beh_data.iloc[rule_shifts_ind[curr_rule]]['rule']
        # check only trials before the first correct
        # find the first correct trial after rule shift
        first_corr_trial = min(beh_data['correct'].iloc[rule_shifts_ind[curr_rule]+1:][beh_data.correct==1].index)

        # Skip the first trial after rule shift because that error is a warning error
        for i in range(rule_shifts_ind[curr_rule]+2,  first_corr_trial):
            if old_rule in beh_data.iloc[i]['chosen']:
                beh_data.at[i, 'preservative_error'] = 1.

    perservative_errors_dict = feature_count(beh_data['preservative_error'])

    # Now we have to check for cross conditions
    rule_feedback_dict = feature_count(beh_data['rule'], beh_data['correct'], double=True)

    rule_dim_feedback_dict = feature_count(beh_data['rule dimension'], beh_data['correct'], double=True)

    enumeration_dict = {'rule_dim_x_feedback': rule_dim_feedback_dict, 'rule_x_feedback': rule_feedback_dict,
                        'perservative_errors': perservative_errors_dict, 'rule': rule_dict, 'feedback':feedback_dict,
                        'rule_dim': rule_dim_dict, 'key_press':key_press_dict, 'warning error': num_problems_dict}
    return enumeration_dict


def main():

    subject = 'IR87'
    print(os.getcwd())

    # This works on Windows, hasn't been tested on Linux/Mac
    data_dir = Path(f"{os.pardir}/data/{subject}")
    sessions = os.listdir(data_dir)
    results_dir = Path(f"{os.pardir}/results")
    global_dictionary = {}
    for session in sessions:
        curr_session = int(session[-1])
        print(curr_session)
        if curr_session in [1,2]:
            continue
        else:
            file_path = data_dir / f"sess-{curr_session}" / "behavior" /  f'sub-{subject}-sess-{curr_session}-beh.csv'

            # Linux
            # data_directory = f"/home/eduardo/WCST_Human/{subject}"
            # sessions = os.listdir(data_directory)
            # curr_session = int(sessions[0][-1])
            # the_rest = f"sess-{curr_session}/behavior"
            # results_directory = "/home/eduardo/tt_su/results"

            # Windows
            beh_data, _, (in_eq, in_shifts) = process_wcst_behavior(file_path)
            enumeration_dict = enumerate_cond(beh_data)
            global_dictionary[curr_session] = enumeration_dict
            # Also add in here, collapsing across sessions


            # Next, we just need to put it all together into a dataframe
    print('hooray')

if __name__ == "__main__":
    main()