# This script is really about taking our pre-existing behavioral analysis and enumerating conditions better


# First import in the other necessary functions
from behavior_analysis import process_wcst_behavior
import os
import numpy as np
from pathlib import Path
from collections import Counter
import pandas as pd


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


def enumerate_cond_sess(beh_data):
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
    print(rule_dict)

    rule_dim_dict = feature_count(beh_data['rule dimension'])

    key_press_dict = feature_count(beh_data['key press'])

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
        if len(beh_data['correct'].iloc[rule_shifts_ind[curr_rule]+1:][beh_data.correct==1].index) == 0:
            continue
        else:
            first_corr_trial = min(beh_data['correct'].iloc[rule_shifts_ind[curr_rule]+1:][beh_data.correct == 1].index)

        # Skip the first trial after rule shift because that error is a warning error
        for i in range(rule_shifts_ind[curr_rule]+2,  first_corr_trial):
            if old_rule in beh_data.iloc[i]['chosen']:
                beh_data.at[i, 'preservative_error'] = 1.

    perservative_errors_dict = feature_count(beh_data['preservative_error'])

    # Now we have to check for cross conditions
    rule_feedback_dict = feature_count(beh_data['rule'], beh_data['correct'], double=True)

    rule_dim_feedback_dict = feature_count(beh_data['rule dimension'], beh_data['correct'], double=True)

    feedback_keypress_dict = feature_count(beh_data['correct'], beh_data['key press'], double=True)

    enumeration_dict = {'rule_dim_x_feedback': rule_dim_feedback_dict, 'rule_x_feedback': rule_feedback_dict,
                        'feedback_x_keypress': feedback_keypress_dict, 'perservative_errors': perservative_errors_dict,
                        'rule': rule_dict, 'feedback':feedback_dict,
                        'rule_dim': rule_dim_dict, 'key_press': key_press_dict, 'warning error': num_problems_dict}
    return enumeration_dict


def enumerate_cond_subject(data_dir, results_dir, subject):
    """
    This code builds on the previous two functions above to enumerate conditions for all sessions within a single
    subject, separately and by collapsing across them. Output is a csv file that is saved to results_dir
    :param data_dir: Path object. Where the data lives. Expected file structure is subject/sess-#/behavior/file.csv
    :param results_dir: Path object. Where to output results.
    :param subject:  String. Subject identifier
    :return:
    """
    print('hello')
    print(data_dir)
    global_dict = {}
    sessions = os.listdir(data_dir)
    for session in sessions:
        curr_session = int(session[-1])
        print(curr_session)
        # this is the one magic string in this code
        files = os.listdir(data_dir / session / "behavior")
        file_name = [file for file in files if file.endswith('.csv')]
        file_path = data_dir / f"sess-{curr_session}" / "behavior" / file_name[0]

        # Linux
        # data_directory = f"/home/eduardo/WCST_Human/{subject}"
        # sessions = os.listdir(data_directory)
        # curr_session = int(sessions[0][-1])
        # the_rest = f"sess-{curr_session}/behavior"
        # results_directory = "/home/eduardo/tt_su/results"

        # Windows
        beh_data, _, _ = process_wcst_behavior(file_path)
        enumeration_dict = enumerate_cond_sess(beh_data)
        global_dict[curr_session] = enumeration_dict
        # Also add in here, collapsing across sessions

    # Next, we just need to collapse data into a bigger summary dictionary to get a sense for what's possible if we
    # we want to do pseudo-population things
    summary_dict = {}
    for dict_key in global_dict[curr_session].keys():
        print(dict_key)
        summary_dict[dict_key] = Counter(global_dict[curr_session][dict_key])
        for session in sessions:
            curr_session = int(session[-1])
            if dict_key not in summary_dict:
                summary_dict[dict_key] = Counter(global_dict[curr_session][dict_key])
            else:
                summary_dict[dict_key] += Counter(global_dict[curr_session][dict_key])

    results_file = results_dir / f"{subject}_summary_stats.xlsx"
    with pd.ExcelWriter(results_file) as writer:
        for session in sessions:
            curr_session = int(session[-1])
            pd.DataFrame(global_dict[curr_session]).to_excel(writer, sheet_name=f"sess-{curr_session}")
        pd.DataFrame(summary_dict).to_excel(writer, sheet_name="Summary")


def main():

    results_dir = Path(f"{os.pardir}/results")
    print(results_dir)
    print('hi again')
    # subjects = ['IR87', 'IR86', 'DA9', 'IR84', 'IR85', 'IR94', 'IR95', 'IR99']
    subjects = ['BERK01','BERK02']
    for subject in subjects:
        print(os.getcwd())
        print(subject)
        # data_dir = Path(f"{os.getcwd()}/wcst_human/data/{subject}/")
        data_dir = Path(f"{os.pardir}/data/{subject}/")
        # This works on Windows, hasn't been tested on Linux/Mac
        # data_dir = Path(f"{os.pardir}/data/{subject}")
        enumerate_cond_subject(data_dir, results_dir, subject)

    print('hooray')

if __name__ == "__main__":
    main()

    