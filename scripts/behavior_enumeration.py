# This script is really about taking our pre-existing behavioral analysis and enumerating conditions better


# First import in the other necessary functions
from behavior_analysis import process_wcst_behavior
import os


def feature_count(feature_list):
    """
    Counts the number of each distinct element in list
    :param feature_list: list of strings, int
    :return: counts_dict: dict.
    """
    features = set(list(feature_list))

    feature_split = [(i, list(feature_list).count(i)) for i in features]
    counts_dict = {}
    counts_dict.update(feature_split)
    return counts_dict

subject = 'IR87'
session = '4'

# Linux
data_directory = f"/home/eduardo/WCST_Human/{subject}/sess-{session}/behavior"
results_directory = "/home/eduardo/tt_su/results"

# Windows
data_directory = f"C:\\Users\edsan\PycharmProjects\wcst_human\data\{subject}\sess-{session}\\behavior"
results = f"C:\\Users\edsan\PycharmProjects\wcst_human\\results"

# next step use the file names to process the behavior and read that in
file_name = f'sub-{subject}-sess-{session}-beh.csv'
file_path = os.path.join(data_directory, file_name)
beh_data, rule_shifts_ind, (in_eq, in_shifts) = process_wcst_behavior(file_path)

print(beh_data)
# next step, enumerate incorrect / correct numbers
corr = len(beh_data[beh_data.correct == 1])
incorr = len(beh_data[beh_data.correct == 0])

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
old_rules = beh_data.iloc[rule_shifts_ind]['rule']
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

# Now we have to check
