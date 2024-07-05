from intracranial_ephys_utils.manual_process import write_timestamps
from pathlib import Path
import os


def main():
    # subjects = ["IR95"]
    # subjects = ["IR87"]
    # subjects = ["IR84"]
    # subjects = ["IR85", "IR94"]
    # subjects = ["IR86"]
    # subjects = ["IR98", "IR99"]
    subjects = ["IR100"]
    sessions = ["sess-1" 'sess-2']
    # sessions = ["sess-1", "sess-2", "sess-3"]
    # sessions = ["sess-4"]
    # sessions = ["sess-1"]
    task = 'wcst'
    for subject in subjects:
        for session in sessions:
            general_directory = Path(f"{os.pardir}/data/{subject}/{session}")
            data_directory = general_directory / "raw"
            results_directory = general_directory
            write_timestamps(subject, session, task, data_directory, general_directory, results_directory)


if __name__ == "__main__":
    main()
