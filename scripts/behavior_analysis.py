
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def get_wcst_rt(file_name):
    beh_data = pd.read_csv(file_name)
    trials = beh_data['trials_thisIndex']
    rt = beh_data['response_time']
    return trials, rt


def process_wcst_behavior(file_name, running_avg=5):
    """
    In progress. Takes raw behavioral data from WCST and adds in features to be regressed later, including
    choice, rule dimension and running average(from hyperparamter). Note the "S" pertains to two different rules.
    To alleviate this issue, this code recasts the texture S rule as W, as in sWirl. The left over Ss pertain to the
    star shape.
    :param file_name: string
        path to the csv file containing wcst behavioral data
    :param running_avg: int, optional
        integer that determines how to calculate the running average accuracy
    :return: beh_data: dataframe
        dataframe containing behavior data for participant
    :return: rule_shifts_ind : list of bool
        the indices (raw) where the rule shifts
    :return: incorrect_eq: tuple
        whether the rule or shift determined on our side, matches the one in the csv file.
    """
    beh_data = pd.read_csv(file_name)
    subject = beh_data['participant'][0]
    session = beh_data['session'][0]

    if 'key_resp_2_keys' in beh_data.columns:
        beh_data = beh_data.rename(columns={'key_resp_2_keys': 'key press',
                                            'key_resp_2_rt': 'rt',
                                            'key_resp_3_keys': 'off trial key press',
                                            'key_resp_3_rt': 'off trial rt',
                                            'trials_thisRepN': 'trials.thisRepN',
                                            'trials_thisN': 'trials.thisN',
                                            'trials_thisIndex': 'trials.thisIndex',
                                            'trials_thisTrialN': 'trials.thisTrialN'})
    else:
        # If WCST version 6, throw warning and rename columns
        print(f'Check that {subject} and {session} were run in the same experiment version as others')
        beh_data = beh_data.rename(columns={'key_resp_2.keys': 'key press',
                                            'key_resp_2.rt': 'rt',
                                            'key_resp_3.keys': 'off trial key press',
                                            'key_resp_3.rt': 'off trial rt',
                                            'shift_type': 'rule_shift',
                                            'Unnamed: 21': 'nothing'})
    # These tell us how the key pressed maps to the image locations
    keys_img_location_dict = {'left': 4, 'down': 2, 'right': 3, 'up': 1,
                              'None': 'None'}

    # Drop columns I don't care about
    beh_data = beh_data.drop(['date', 'frameRate', 'trials.thisRepN', 'trials.thisN',
                              'trials.thisIndex', 'off trial key press', 'off trial rt'], axis=1)

    beh_data = beh_data.rename(columns={'trials.thisTrialN': 'trial'})
    # Rule is just a letter, but I can match it to rule dimension if it's correct,
    # note though that 's' was coded twice, so I have to double back with these...
    # Note that these 's's may still be wrong if for some reason the rule changes from 's' to 's'.
    # So if the rule is S it could be shape or texture, who's to say.

    # NOTE: The original experiment behavioral files have a way to tell between different s's. If we can find those
    # files, this stops not working
    rule_dict = {'S': 'Problem', 'T': 'Shape', 'C': 'Shape', 'Q': 'Shape', 'B': 'Color', 'Y': 'Color', 'G': 'Color',
                 'M': 'Color', 'L': 'Texture', 'P': 'Texture', 'W': 'Texture', 'R': 'Texture'}
    # Cover every possible version of this experiment
    rule_shift_dict = {'yes': True, 'no': False, True: True, False: False, '0': False, 'extra': True, 'intra': True}

    # Here we're going to process the data itself
    problem_rows = []
    beh_data['correct'] = int(0)
    beh_data['chosen'] = ''
    beh_data['rule_shift_bool'] = ''
    beh_data[f'running_avg_{running_avg}'] = 0.
    beh_data['rule dimension'] = ''
    beh_data['correct card'] = ''

    for row in beh_data.index.values:
        if pd.isna(beh_data.loc[row, 'key press']):
            continue
        resp = keys_img_location_dict[beh_data.loc[row, 'key press']]
        if not pd.isna(beh_data.loc[row, 'rule']):
            rule = beh_data.loc[row, 'rule'].strip()
            corr_card = [i.replace('.bmp', '') for i in
                         list(beh_data.loc[row, ['bmp_table_1', 'bmp_table_2', 'bmp_table_3', 'bmp_table_4']])
                         if (rule in i) and (i[beh_data.loc[row, 'rule'].index(rule)] == rule)]
            if beh_data.loc[row, 'rule'].strip() == 'S':
                rule_ind = beh_data.loc[row, 'rule'].index('S')
                if rule_ind == 2:
                    rule = 'W'
                else:
                    rule = 'S'
        beh_data.loc[row, 'rule_shift_bool'] = rule_shift_dict[
            beh_data.loc[row, 'rule_shift']]
        beh_data.loc[row, 'rule dimension'] = rule_dict[rule]
        if len(corr_card) > 1:
            print('Our code for finding the correct card has duplicate cards')
        else:
            beh_data.loc[row, 'correct card'] = corr_card[0]

        if resp == 'None':
            beh_data.loc[row, 'chosen'] = 'None'
        else:
            beh_data.loc[row, 'chosen'] = beh_data.loc[
                row, f'bmp_table_{resp}'].replace('.bmp', '')

        beh_data.loc[row, 'rule'] = rule
            # problem_rows.append(row)
        if rule in beh_data.loc[row, 'chosen']:
            beh_data.loc[row, 'correct'] = int(1)
        # magic number 5 bc running average is 5
        beh_data.loc[row, f'running_avg_{running_avg}'] = np.mean(
            beh_data.loc[np.arange(max(row - running_avg + 1, 0), row + 1), 'correct'])

    # # Hopefully irrelevant
    # # The rule S can mean two separate rules. So we'll check for this.
    # if len(problem_rows) > 0:
    #     # General solution for figuring out what double coded is, in an algorithmic way:
    #     # The algorithm is to find long runs of S, assume it's one rule and rules don't switch from S to S.
    #     # For these long runs, unless it's the end of the task, the last five trials of the rule should be 100% accurate,
    #     # or at least 3/5. For these correct ones, look at what image was chosen and check the index. Since sometimes,
    #     # S can appear in both, we just take the mode of the indexes over these correct trials.
    #
    #     # Check for discontinuities (indicates different rules)
    #     sorted_problem_rows = np.sort(problem_rows)
    #     problem_rows_diff = np.abs(np.diff(sorted_problem_rows))
    #     # This logic is a bit convoluted but stay with me here. I'm going to take all the problem rows indices, and then
    #     # sort them. From there, look for cases where the index changes more than one (discontinuity).
    #     # From here, we assume that this discontinuity is greater than 5 to truly be separate problems.
    #
    #     actual_index = np.argwhere(problem_rows_diff>1)
    #     if actual_index.shape[0] == 0:
    #         s_in_chosen = [beh_data.loc[row, 'chosen'].index('S') for row in
    #                        sorted_problem_rows
    #                        if (len(re.findall('S', beh_data.loc[row, 'chosen'])) == 1 and beh_data.loc[
    #                 row, 'ans_correctness'] > 0)]
    #         index_mode = stats.mode(s_in_chosen, keepdims=False)[0]
    #         if index_mode == 0:
    #             beh_data.loc[problem_rows, 'rule dimension'] = 'Shape'
    #         else:
    #             beh_data.loc[problem_rows, 'rule dimension'] = 'Texture'
    #         print(f'mode:{index_mode}')
    #     else:
    #         raise NotImplementedError
    #         problem_diff_ind = problem_rows_diff[problem_rows_diff > 1]
    #         s_in_chosen = [beh_data.loc[row, 'chosen'].index('S') for row in sorted_problem_rows[:actual_index[0][0]]
    #                        if (len(re.findall('S', beh_data.loc[row, 'chosen'])) == 1 and beh_data.loc[
    #                 row, 'ans_correctness'] > 0)]

    # Check for internal consistency because one of the csvs have weird rule shift parameters, that don't match the
    # rest of the file
    incorrect_eq = (beh_data['correct'] != beh_data['ans_correctness']).any()
    rule_shifts_ind = list(beh_data[beh_data.rule_shift_bool == True]['trial'].astype(int))
    beh_data.dropna(subset='key press', inplace=True)
    return beh_data, rule_shifts_ind, incorrect_eq


def wcst_features(file_name):
    beh_data, _, _ = process_wcst_behavior(file_name, 5)
    trials = beh_data['trial']
    chosen = beh_data['chosen']
    rule = beh_data['rule']
    feedback = beh_data['correct']
    rule_dimension = beh_data['rule dimension']
    return trials, chosen, rule, feedback, rule_dimension


def plot_subject_performance(trial_num, corrects, rule_shifts, subject,
                             session, output_folder=None):
    """

    :param trial_num:
    :param corrects: Running average correctness over trials
    :param rule_shifts:
    :param subject:
    :param session:
    :param output_folder:
    :return:
    """
    # We expect dataframe series for the first two and a 
    # list of dataframe indexes for the third argument
    fig = plt.figure()
    plt.plot(trial_num, corrects, linestyle='dashed')
    plt.vlines(rule_shifts[rule_shifts == True].index, 1, 0, 'red')
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
    """

    :param summary_data_df:
    :param output_folder:
    :return:
    """
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
