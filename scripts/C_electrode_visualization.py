import mne
import numpy as np
from mne.io import read_raw_fif
from mne.datasets import fetch_fsaverage
import pandas as pd
import os
from BB_processlfp import lfp_prep

test_subject = 'IR95'

# input data
subjects_dir = "*/mne_data/MNE-fsaverage-data"
anat_csv = f"{os.pardir}/data/{test_subject}/{test_subject}_Elec_Notes.xlsx"
# anat_csv = "/path/to/csv/file/with/mni/coordinates/coords_mni.csv"
# raw = read_raw_fif("/path/to/seeg/sub-009_preproc_ieeg.fif")  # preprocessed seeg data

# epoch at start of audio playback
# epochs = mne.Epochs(raw, event_id={"wav_playback"}, detrend=1, baseline=None)

# load the coordinate values
coords = pd.read_excel(anat_csv)

# 3 coordinates in one cell, split them and cast them to new channels
coordinate_data = [list(coords.loc[i, 'Coordinates'].split()) for i in coords.index.values]
coords['MNI_1'] = np.array(coordinate_data)[:, 0]
coords['MNI_2'] = np.array(coordinate_data)[:, 1]
coords['MNI_3'] = np.array(coordinate_data)[:, 2]

test_session = 'sess-3'
task = 'wcst'
bp = 1000
event_lock = 'Feedback'
feature = 'correct'
# feature = 'Feedback'
standardized_data = True
# regularization_setting = 'auto'
regularization_setting = None
electrode_selection = 'all'
# baseline = (2, 2.5)
baseline = (-0.5, 0)
car_setting = False
epochs_dataset, trial_time, microwire_names, feature_values = lfp_prep(test_subject, test_session,
                                                                       task, event_lock=event_lock,
                                                                       feature=feature,
                                                                       baseline=baseline, smooth=True,
                                                                       electrode_selection=electrode_selection,
                                                                       car=car_setting)

coords = coords[["Electrode", "MNI_1", "MNI_2", "MNI_3", "Loc Meeting"]]
# coords = coords[coords.Electrode.isin(microwire_names)]
# epochs = epochs["Response"][0]  # just process one epoch of data for speed
montage = epochs_dataset.get_montage()
from mne.datasets import fetch_fsaverage

# paths to mne datasets - sample sEEG and FreeSurfer's fsaverage subject
# which is in MNI space
misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()
subjects_dir = sample_path / "subjects"

# use mne-python's fsaverage data
fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)  # downloads if needed
# first we need a head to mri transform since the data is stored in "head"
# coordinates, let's load the mri to head transform and invert it
this_subject_dir = misc_path / "seeg"
# head_mri_t = mne.coreg.estimate_head_mri_t("sample_seeg", this_subject_dir)
# # apply the transform to our montage
# montage.apply_trans(head_mri_t)
#
# # now let's load our Talairach transform and apply it
# mri_mni_t = mne.read_talxfm("sample_seeg", misc_path / "seeg")
# montage.apply_trans(mri_mni_t)  # mri to mni_tal (MNI Taliarach)
#
# # for fsaverage, "mri" and "mni_tal" are equivalent and, since
# # we want to plot in fsaverage "mri" space, we need use an identity
# # transform to equate these coordinate frames
# montage.apply_trans(mne.transforms.Transform(fro="mni_tal", to="mri", trans=np.eye(4)))

# epochs_dataset.set_montage(montage)
# # compute the transform to head for plotting
# trans = mne.channels.compute_native_head_t(montage)
# # note that this is the same as:
# # ``mne.transforms.invert_transform(
# #      mne.transforms.combine_transforms(head_mri_t, mri_mni_t))``
#
# view_kwargs = dict(azimuth=105, elevation=100, focalpoint=(0, 0, -15))
# brain = mne.viz.Brain(
#     "fsaverage",
#     subjects_dir=subjects_dir,
#     cortex="low_contrast",
#     alpha=0.25,
#     background="white",
# )
# brain.add_sensors(epochs_dataset.info, trans=trans)
# brain.add_head(alpha=0.25, color="tan")
# brain.show_view(distance=400, **view_kwargs)

# # convert channel coordinate values to meters
# coords[["MNI_1", "MNI_2", "MNI_3"]] /= 1000
#  # drop micro channels (labels start with `u`), not interested in these
# coords = coords.loc[~coords["Channels"].str.contains("u")]




# SUBJECTS_DIR = "*/mne_data/MNE-fsaverage-data"

# select left/right
# (see `coords` data frame in first message)
left_chans = coords.Electrode.str.startswith("L")
right_chans = coords.Electrode.str.startswith("R")

# convert to (n_elecs, 3) arrays
xyz_left = coords.loc[left_chans, ["MNI_1", "MNI_2", "MNI_3"]].to_numpy()
xyz_right = coords.loc[right_chans, ["MNI_1", "MNI_2", "MNI_3"]].to_numpy()

brain = mne.viz.Brain(
    subject="fsaverage",
    cortex="low_contrast",
    hemi='both',
    alpha=0.25,
    background="white",
    subjects_dir=subjects_dir,
    size=(1600, 800),
    show=True,
)

color_dict = {'BLA':'red','CA1':'red', 'AC':'purple','OFC':'blue','MFG':'green','MTG':'yellow','STS':'orange',
              'STG':'pink', 'white matter': 'black', 'OoB':'white'}

# color_dict = {'BLA':'red'}
colors = [color_dict[coords.loc[i,'Loc Meeting']] for i in coords.index.values]

for i, xyz in enumerate(xyz_left):
    brain.add_foci(
        xyz,
        hemi="lh",
        color=colors[i],
        scale_factor=0.3
    )

# brain.add_foci(
#     xyz_right,
#     hemi="rh",
#     color=colors,
#     scale_factor=0.3
# )
brain.save_image(f'{os.pardir}/results/sample_brain_sagittal.png')

brain.show_view('rostral')
brain.save_image(f'{os.pardir}/results/sample_brain_rostral.png')
brain.show_view('axial')
brain.save_image(f'{os.pardir}/results/sample_brain_axial.png')
