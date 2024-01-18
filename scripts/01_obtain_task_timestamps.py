from intracranial_ephys_utils.load_data import read_task_ncs
import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
from behavior_analysis import get_wcst_rt


def diagnostic_time_series_plot(lfp_signal, freq, output_folder=None, electrode_name='', total_time=None,
                                subject=None, session=None):
    """
    This script serves to plot basically any LFP signal for sanity check, but without interactivity like
    the dynamic script.
    :param lfp_signal: array. This should 1Xn_timepoints
    :param freq: (float)
    :param output_folder: (Path). Optional. Useful for saving snippers
    :param electrode_name: (string). Optional. Useful for writing title to plot
    :param total_time: (string). Optional. Useful for plotting subsets of experiment
    :param subject: (string). Subject identifier.
    :param session: (string). Session identifier.
    :return:
    """
    fig2, ax = plt.subplots(4, 1)
    ax[0].plot(np.linspace(0, 1, num=int(freq)), lfp_signal[0:int(freq)])
    midlevel_time = 30
    ax[1].plot(np.linspace(0, midlevel_time, num=int(freq*midlevel_time)), lfp_signal[0:int(midlevel_time*freq)])
    sec_in_minute = 60
    ax[2].plot(np.linspace(0, lfp_signal.shape[0]/freq/sec_in_minute, lfp_signal.shape[0]), lfp_signal)
    ax[3].plot(np.linspace(midlevel_time, 0, num=int(freq*midlevel_time)), lfp_signal[-int(midlevel_time*freq):])
    for axis in ax:
        axis.set_ylabel('Voltage (uV)')
        axis.set_xlabel('Time (s)')
    ax[2].set_xlabel('Time (m)')
    if total_time is not None:
        ax[2].set_xlim([0, total_time])
    if electrode_name != '':
        plt.suptitle(f'Time Courses for {electrode_name}')
    if output_folder is not None:
        plt.savefig(os.path.join(output_folder, f'{subject}_{session}.png'))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def ph_bhv_alignment(ph_signal, trials, rt, sample_rate, cutoff_fraction=4, task_time=None):
    '''
    Script processes photodiode signal. Assumes 1 event, and requires task time to limit analysis to.
    :param ph_signal: This is the photodiode signal itself (array of floats)
    :param trials:  This is the behavioral data trial number (array filled with integers)
    :param rt:  This is the behavioral data response times for each trial (array filled with floats, units of s)
    :param sample_rate: How many samples per second (float)
    :param cutoff_fraction: Fraction that determines the cutoff between 0 and 1 for photodiode signal
    :param task_time: This is how long the task took (in seconds). Helpful to zoom in on particular data regions.
            optional (float)
    :return: event_lengths_pruned: Array of event_lengths in seconds, to be plotted with the rts (array of floats)
    :return: ph_onset_trials: Array of event onsets in samples (array of int)
    :return: ph_offset_trials: Array of event offsets in samples (array of int)
    :return: padded_rts:
    '''
    if task_time is not None:
        total_time = int(task_time*sample_rate*60)
    else:
        total_time = len(ph_signal)

    midpoint = (max(ph_signal[0:total_time])-min(ph_signal[0:total_time]))/cutoff_fraction+min(ph_signal[0:total_time])
    ph_signal_bin = np.zeros((total_time, ))
    ph_signal_bin[ph_signal[0:total_time] > midpoint] = 1.

    diagnostic_time_series_plot(ph_signal_bin, sample_rate, electrode_name='Binary Photodiode')

    onset_events = np.where(np.diff(ph_signal_bin) > 0)[0]
    offset_events = np.where(np.diff(ph_signal_bin) < 0)[0]
    print(onset_events.shape)
    print(offset_events.shape)
    ideal_trial_number = np.arange(max(trials))
    min_rt = np.nanmin(rt)  # min but ignore faulty key press (logged as nan)

    # We might start with the photodiode on!
    if len(offset_events) > len(onset_events) and offset_events[0] < onset_events[0]:
        offset_events = offset_events[1:]
    if len(onset_events) > len(offset_events) and onset_events[-1] > offset_events[-1]:
        onset_events = onset_events[:-1]


    print(f"onset check {onset_events.shape}")
    print((offset_events-onset_events) / sample_rate)
    # why use onset for both? because difference between onset and offset isn't constant so there may be variable number
    # of events with difference greater than some value across onset and offset
    # This is going to get rid of events that are under some specific time, here the 70% of minimum response time
    event_cutoff = 0.7
    onset_events_pruned = onset_events[np.diff(onset_events,
                                               append=onset_events[-1]+(sample_rate*min_rt*event_cutoff))
                                       >= sample_rate * min_rt * event_cutoff]
    offset_events_pruned = offset_events[np.diff(onset_events,
                                                 append=onset_events[-1]+(sample_rate*min_rt*event_cutoff))
                                         >= sample_rate * min_rt * event_cutoff]

    event_lengths_pruned = (offset_events_pruned-onset_events_pruned) / sample_rate
    trials = np.array(trials)
    # loop to add variable numbers of nan to pad rt
    # double check for starting trials
    start_trial_num = np.sort(trials)[0]
    if start_trial_num > 0:
        # padded_trials = np.insert(trials, 0, np.arange(start_trial_num)+1)
        padded_rts = np.insert(np.array(rt), 0, np.arange(start_trial_num) + 1)
    else:
        # padded_trials = trials
        padded_rts = np.array(rt)

    # np.diff makes a new array of n-1, I'm adding a nan, so I can index easier
    trial_diff_num = np.diff(trials, append=np.array(np.NAN))
    missing_trials_num = trials[trial_diff_num > 1]
    missing_trials_ind = trial_diff_num[trial_diff_num > 1]
    print(missing_trials_num)
    print(missing_trials_ind)
    for ind, trial_num in enumerate(np.sort(missing_trials_num)):
        # rationale is this
        # the index tells us the difference in trials, so we pad it with missing trials,
        # but we pad it with one less than the difference, and also account for indexing at 0
        # ... sorry
        missing_num = int(missing_trials_ind[ind])
        # # use below for testing this code!
        # padded_trials = np.insert(padded_trials, trial_num+1, np.arange(1, missing_num) + trial_num)
        # print(padded_trials)
        padded_rts = np.insert(padded_rts, trial_num+1, np.full((missing_num-1,), np.nan))

    # Onset and offset in samples
    # max_event_length = 3.97
    max_event_length = 4.5
    # 0.9 for IR87 sess-1 and sess-2
    min_rt_factor = 0.7
    print(f"onset check {onset_events_pruned.shape}")
    print(event_lengths_pruned)
    ph_onset_trials = onset_events_pruned[(event_lengths_pruned > min_rt*min_rt_factor) & (event_lengths_pruned <= max_event_length)]
    ph_offset_button_press = offset_events_pruned[(event_lengths_pruned > min_rt*min_rt_factor) & (event_lengths_pruned <= max_event_length)]

    event_lengths_pruned = event_lengths_pruned[(event_lengths_pruned > min_rt*min_rt_factor) & (event_lengths_pruned <= max_event_length)]
    print(max(trials))
    print(padded_rts.shape)
    print(event_lengths_pruned)
    return event_lengths_pruned, ph_onset_trials, ph_offset_button_press, padded_rts


def photodiode_behavior_alignment_plot(padded_rts, event_lengths_pruned, output_folder=None, subject=None,
                                       session=None):
    """
    Plots event lengths and rts for WCST to ensure photodiode detection behaves as expected
    :param padded_rts:
    :param event_lengths_pruned:
    :param output_folder:
    :param subject:
    :param session:
    :return:
    """
    plt.plot(padded_rts, label='response time')
    plt.plot(event_lengths_pruned, linestyle='dashed', alpha=0.7, label='photodiode event time')
    plt.xlabel('Trial Number')
    plt.ylabel('Event Length (s)')
    plt.title(f'{subject} - session {session} \n alignment with behavior')
    plt.legend()
    if output_folder is not None:
        plt.savefig(os.path.join(output_folder, f'{subject}_{session}.png'))
        plt.close()
    else:
        plt.show()


def check_events(ph_signal, sample_rate, ph_onset_trials, tstart=None, tend=None):
    '''
    Simple plot with detected onset events overlaid photodiode signal
    :param ph_signal:
    :param ph_onset_trials:
    :param tstart:
    :param tend: in seconds (optional)
    :return:
    '''
    if tend is None:
        tend = len(ph_signal)
        tstart = int(tend - 30*sample_rate)
    else:
        tend = int(tend*sample_rate)
        tstart = int(tstart*sample_rate)
    ph_signal_pruned = ph_signal[tstart: tend]
    plt.plot(ph_signal[tstart: tend])
    plt.vlines(ph_onset_trials[(ph_onset_trials > tstart) & (ph_onset_trials < tend)]-tstart,
               max(ph_signal_pruned), min(ph_signal_pruned), 'red')
    plt.show()


def main():
    test_subject = 'IR87'
    test_session = 'sess-4'
    task = 'wcst'
    folder_name = Path(f'{os.pardir}/data/{test_subject}/{test_session}/raw')
    event_file = folder_name.parents[0] / f"{test_subject}_{test_session}_{task}_events.csv"
    ph_file_path = 'photo1.ncs'
    split_tup = os.path.splitext(ph_file_path)
    filename = split_tup[0]
    ph_signal, sample_rate, _, _ = read_task_ncs(folder_name, ph_file_path, task=task,
                                                 events_file=event_file)

    diagnostic_time_series_plot(ph_signal, sample_rate, electrode_name=filename)

    bhv_file_path = folder_name.parents[0] / "behavior" / f"sub-{test_subject}-{test_session}-beh.csv"
    # Now to get ground truth data for wisconsin card sorting
    trials, rt = get_wcst_rt(bhv_file_path)

    # Use both behavioral data and photodiode data to do alignment
    # remember that task_start_segment_time is the baseline for the array
    event_lengths_pruned, ph_onset_trials, ph_offset_button_press, padded_rts = ph_bhv_alignment(ph_signal, trials,
                                                                                                 rt, sample_rate,
                                                                                                 cutoff_fraction=2)
    check_events(ph_signal, sample_rate, ph_onset_trials)

    photodiode_behavior_alignment_plot(padded_rts, event_lengths_pruned, subject=test_subject, session=test_session)

    # Okay now with our photodiode signal in tow, we'll get a histogram
    plt.hist(ph_signal)
    plt.show()

    # Next we save our exported photodiode detected events, matched with wcst rts
    beh_time_data = np.array([ph_onset_trials, ph_offset_button_press, padded_rts])
    beh_df = pd.DataFrame(beh_time_data.T, columns=['Onset', 'Feedback', 'RTs'])
    beh_df.to_csv(folder_name.parents[0] / "behavior" / f"sub-{test_subject}-{test_session}-ph_timestamps.csv")


if __name__ == "__main__":
    main()