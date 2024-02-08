from intracranial_ephys_utils.load_data import get_event_times
from pathlib import Path


def write_timestamps(event_folder, sort_folder):
    """
    Looks in event folders for labels. Elicits user input to determine which labels are relevant for spike sorting
    to constrain looking at only task-relevant data. User can input -1 if the whole datastream should be spike sorted.
    :param event_folder: This is the folder where the event file is
    :param sort_folder:  This is where the microwire data to be sorted is
    :return: None. A txt file is generated with relative timestamps if needed, or not if not needed.
    """
    event_times, event_labels, global_start = get_event_times(event_folder, rescale=False)
    event_times_sec, _, _ = get_event_times(event_folder, rescale=True)
    print(event_labels)
    task_start = int(input("Where does the task start? (start counting at 0), -1 if we should take the whole file"))

    if task_start < 0:
        return None
    else:
        buffer = 30  # buffer of 30 seconds before and after task annotation in case of experimenter delay
        optimal_start = event_times_sec*10**-6-global_start-buffer
        if optimal_start < global_start:
            spike_sort_start = global_start
        else:
            spike_sort_start = optimal_start
        spike_sort_end = event_times[task_start+1]*10**-6+buffer
    timestamps_file = sort_folder / f"timestampsInclude.txt"
    with open(timestamps_file, 'w') as f:
        f.write(f'{spike_sort_start}    {spike_sort_end}')
    return None


def main():
    event_folder = Path("C:\\Users\\edsan\\PycharmProjects\\tt_su\\data\\WCST_Human\\IR87\\sess-4\\raw")
    results_folder = Path("C:\\home\\knight\\sandoval\\spike_sorting\\IR87\\tt\\sess-2\\")
    write_timestamps(event_folder, results_folder)

if __name__ == "__main__":
    main()