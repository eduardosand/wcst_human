
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy import stats
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:52:27 2023

@author: edsan

The aim of this script is to figure out the top 50% of participants
that completed the wisconsin card sorting
To do that, we need the following:
    
    1. Load in data from one participant
    To get response measures, we need to link their response (arrow keys) to
    the image they chose. If the image contains the rule, we count as correct
    
    2. Plot responses overlaid on top of the correct responses
    
    3. Obtain two course grain measures. One is raw accuracy and the other is
    how many problems were solved
    
    4. Place both measures, alongside how many trials were missed into a
    separate csv file.
    
    5. For now, stick to one session. If we get all this done, figure out
    how to extend to multiple sessions.
"""


def get_wcst_rt(file_name):
    beh_data = pd.read_csv(file_name)
    trials = beh_data['trials_thisIndex']
    rt = beh_data['response_time']
    return trials, rt


def process_wcst_behavior(file_name, running_avg=5):
    """
    takes behavior in csv files and recodes the rule, choice, rule dimension,
    and running average accuracy based on a hyperparameter
    :param file_name: string
        path to the csv file containing wcst behavioral data
    :param running_avg: int, optional
        integer that determines how to calculate the running average accuracy
    :return: beh_data: dataframe
        dataframe containing behavior data for participant
    :return: rule_shifts_ind : list of bool
        the indices (raw) where the rule shifts
    :return: (incorrect_eq, incorrect_shifts): tuple
        whether the rule or shift determined on our side, matches the one in the csv file.
    """
    beh_data = pd.read_csv(file_name)
    keys_img_location_dict = {'left': 4, 'down': 2, 'right': 3, 'up': 1,
                              'None': 'None'}
    # Rule is just a letter but I can match it to rule dimension if it's correct,
    # note though that 's' was coded twice so I have to double back with these...
    # Note that these 's's may still be wrong if for some reason the rule changes from 's' to 's'. Thanks a lot to
    # whoever made this design decision. So if the rule is S it could be shape or texture, who's to say.
    rule_dict = {'S': 'Problem', 'T': 'Shape', 'C': 'Shape', 'Q': 'Shape', 'B': 'Color', 'Y': 'Color', 'G': 'Color',
                 'M': 'Color', 'L': 'Texture', 'P': 'Texture', 'R': 'Texture'}
    problem_rows = []
    rule_shift_dict = {'yes': True, 'no': False, True: True, False: False}
    beh_data['correct'] = int(0)
    beh_data['chosen'] = ''
    beh_data['rule_shift_bool'] = ''
    beh_data[f'running_avg_{running_avg}'] = 0
    beh_data['rule dimension'] = ''
    for row in beh_data.index.values:
        if pd.isna(beh_data.loc[row, 'key_resp_2_keys']):
            continue
        resp = keys_img_location_dict[beh_data.loc[row, 'key_resp_2_keys']]
        beh_data.loc[row, 'rule_shift_bool'] = rule_shift_dict[
            beh_data.loc[row, 'rule_shift']]
        beh_data.loc[row, 'rule dimension'] = rule_dict[beh_data.loc[row, 'rule']]
        if resp == 'None':
            beh_data.loc[row, 'chosen'] = 'None'
        else:
            beh_data.loc[row, 'chosen'] = beh_data.loc[
                row, f'bmp_table_{resp}'].replace('.bmp', '')
        if beh_data.loc[row, 'rule'].strip() == 'S':
            problem_rows.append(row)
        if beh_data.loc[row, 'rule'].strip() in beh_data.loc[row, 'chosen']:
            beh_data.loc[row, 'correct'] = int(1)
        # magic number 5 bc running average is 5
        beh_data.loc[row, f'running_avg_{running_avg}'] = np.mean(
            beh_data.loc[np.arange(max(row - running_avg + 1, 0), row + 1), 'correct'])
    # The rule S can mean two separate rules. So we'll check for this.
    if len(problem_rows) > 0:
        # General solution for figuring out what double coded is, in an algorithmic way:
        # The algorithm is to find long runs of S, assume it's one rule and rules don't switch from S to S.
        # For these long runs, unless it's the end of the task, the last five trials of the R should be 100% accurate,
        # or at least 3/5. For these correct ones, look at what image was chosen and check the index. Since sometimes,
        # S can appear in both, we just take the mode of the indexes over these correct trials.

        sorted_problem_rows = np.sort(problem_rows)
        problem_rows_diff = np.abs(np.diff(sorted_problem_rows))
        # This logic is a bit convoluted but stay with me here. I'm going to take all the problem rows indices, and then
        # sort them. From there, look for cases where the index changes more than one (discontinuity).
        # From here, we assume that this discontinuity is greater than 5 to truly be separate problems.

        problem_rows_diff_ind = problem_rows_diff[problem_rows_diff > 1]
        # Okay we have indices for when indices skip, so we can take the rows between them as runs of s,
        # we then need to take the index of the last five in a run
        start_with_rule_s = True if problem_rows[0] == 0 else False
        if start_with_rule_s:
            raise NotImplementedError
            # TO DO, this code has never been run, so don't write it until needed...
            # dimension_index = 0
        else:
            if len(problem_rows_diff_ind) == 0:
                s_in_chosen = [beh_data.loc[row, 'chosen'].index('S') for row in sorted_problem_rows
                               if (len(re.findall('S', beh_data.loc[row, 'chosen'])) == 1 and beh_data.loc[row, 'ans_correctness'] >0)]
                index_mode = stats.mode(s_in_chosen, keepdims=False)[0]
                if index_mode == 0:
                    beh_data.loc[problem_rows, 'rule dimension'] = 'Shape'
                else:
                    beh_data.loc[problem_rows, 'rule dimension'] = 'Texture'

    # Check for internal consistency because one of the csvs have weird rule shift parameters, that don't match the
    # rest of the file
    incorrect_eq = (beh_data['correct'] != beh_data['ans_correctness']).any()
    beh_data['rule_shift_data'] = beh_data['rule'] != beh_data['rule'].shift(-1).fillna(beh_data['rule'])
    incorrect_shifts = (beh_data['rule_shift_bool'] != beh_data['rule_shift_data']).any()
    rule_shifts_ind = list(beh_data[beh_data['rule_shift_data']]['trials_thisIndex'])
    return beh_data, rule_shifts_ind, (incorrect_eq, incorrect_shifts)


def wcst_features(file_name):
    beh_data, _, _ = process_wcst_behavior(file_name, 5)
    trials = beh_data['trials_thisIndex']
    chosen = beh_data['chosen']
    rule = beh_data['rule']
    feedback = beh_data['correct']
    rule_dimension = beh_data['rule dimension']
    return trials, chosen, rule, feedback, rule_dimension


def plot_subject_performance(trial_num, corrects, rule_shifts, subject,
                             session, output_folder=None):
    # We expect dataframe series for the first two and a 
    # list of dataframe indexes for the third argument
    fig = plt.figure()
    plt.plot(trial_num, corrects, linestyle='dashed')
    plt.vlines(rule_shifts, 1, 0, 'red')
    plt.xlabel('Trial Number')
    plt.ylabel(f'Accuracy \n (Running Average)')
    plt.title(f'{subject} WCST performance: session {session}')
    if output_folder is not None:
        plt.savefig(os.path.join(output_folder, f'{subject}_{session}.png'))
        plt.close()
    else:
        plt.show()
    return None


def plot_group_performance(summary_data_df, output_folder=None):
    summary_data_df.sort_values("Number of Problems", inplace=True)
    # fig = plt.figure()
    fig, ax = plt.subplots(2, 1)
    x_label = [f"{summary_data_df.loc[row, 'Subjects']}-{summary_data_df.loc[row, 'Session']}" for row in
               summary_data_df.index.values]
    ax[0].bar(x_label, summary_data_df['Number of Problems'])
    ax[1].bar(x_label, summary_data_df['Raw Accuracy'])
    ax[0].set_ylabel('Number of Problems')
    ax[1].set_ylabel('Raw Accuracy')
    ax[0].set_xticks(np.arange(len(x_label)), x_label, fontsize=5, rotation=45)
    ax[1].set_xticks(np.arange(len(x_label)), x_label, fontsize=5, rotation=45)
    plt.title('Overall Behavioral Performance')
    plt.tight_layout()
    if output_folder is not None:
        plt.savefig(os.path.join(output_folder, f'summary.png'))
        plt.close()
    else:
        plt.show()


def main():
    summary_directory = "/home/eduardo/WCST_Human/Summary/U19_WCST_Progress.csv"
    results_directory = "/home/eduardo/tt_su/results"
    summary_df = pd.read_csv(summary_directory)
    summary_data = []
    for row in summary_df.index.values:
        if summary_df.loc[row, 'Sessions'] == 'None':
            continue
        else:
            subject = summary_df.loc[row, 'Subjects']
            sessions = summary_df.loc[row, 'Sessions']
            for session in np.arange(1, int(sessions)+1):
                # The underscore looks weird, but that's what the data on nas is
                file_name = f'sub-{subject}-sess-{session}-beh.csv'
                running_avg = 5
                data_directory = f"/home/eduardo/WCST_Human/{subject}/sess-{session}/behavior"
                file_path = os.path.join(data_directory, file_name)
                print(data_directory)
                beh_data, rule_shifts_ind, _ = process_wcst_behavior(file_path, running_avg)
                plot_subject_performance(beh_data['trials_thisIndex'],
                                         beh_data[f'running_avg_{running_avg}'],
                                         rule_shifts_ind, subject, session,
                                         output_folder=results_directory)

                # df data style subject, session, accuracy, rule shifts
                summary_data.append((subject, session, np.mean(beh_data['correct']),
                                     sum(beh_data['rule_shift_data'])))
    summary_data_df = pd.DataFrame(summary_data, columns=['Subjects', 'Session', 'Raw Accuracy', 'Number of Problems'])

    plot_group_performance(summary_data_df, output_folder=results_directory)


if __name__ == "__main__":
    main()
