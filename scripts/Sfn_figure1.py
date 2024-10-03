# Goal of this part is to generate the dPCA plots that are described in my abstract, but updated
# to account for more recent data cleaning
from G_dPCA import dpca_plot_analysis
from BB_processlfp import organize_data, lfp_prep

# plan
# this code will load in the dataset we downloaded, and attempt to fit a simple dPCA model using just the feedback
# signal

# we will generate three figures, one for each modality
# we will always standardized our datasets
# we will use either baseline
# we will not use any regularization

test_subject = 'IR95'
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
baseline = (2, 2.5)
# baseline = (-0.5, 0)
car_setting = False
epochs_dataset, trial_time, microwire_names, feature_values = lfp_prep(test_subject, test_session,
                                                                       task, event_lock=event_lock,
                                                                       feature=feature,
                                                                       baseline=baseline, smooth=True,
                                                                       electrode_selection=electrode_selection,
                                                                       car=car_setting)
organized_data_mean, organized_data, feedback_dict = organize_data(epochs_dataset, feature_values,
                                                                   standardized_data=standardized_data,
                                                                   method='dPCA')

feature_dict = feedback_dict
feature_name='Feedback'
# suptitle=f'All microwires, bandpassed at {bp}, {lock}-locked'
# plot_signal_avg(organized_data_mean, test_subject, test_session, trial_time, labels=feedback_dict,
#                     extra_string=f'Normalization = {standardized_data} {event_lock}-lock',
#                 signal_names=microwire_names)
if electrode_selection == 'all':
    data_modality_string = 'Macrocontact and Microwire broadband LFP'
else:
    data_modality_string = f'{electrode_selection} broadband LFP'
dpca_1, Z_1 = dpca_plot_analysis(organized_data_mean, organized_data, trial_time, feature_dict, test_subject,
                                 test_session, event_lock, standardized_data,
                                 regularization_setting=regularization_setting,
                                 feature_names=[feature_name], data_modality=data_modality_string)

electrode_selection = 'microwire'
epochs_dataset, trial_time, microwire_names, feature_values = lfp_prep(test_subject, test_session,
                                                                       task, event_lock=event_lock,
                                                                       feature=feature,
                                                                       baseline=baseline, smooth=True,
                                                                       electrode_selection=electrode_selection,
                                                                       car=car_setting)
organized_data_mean_microwire, organized_data_microwire, feedback_dict = organize_data(epochs_dataset, feature_values,
                                                                   standardized_data=standardized_data,
                                                                   method='dPCA')

feature_dict = feedback_dict
# suptitle=f'All microwires, bandpassed at {bp}, {lock}-locked'
# plot_signal_avg(organized_data_mean, test_subject, test_session, trial_time, labels=feedback_dict,
#                     extra_string=f'Normalization = {standardized_data} {event_lock}-lock',
#                 signal_names=microwire_names)
dpca_2, Z_2 = dpca_plot_analysis(organized_data_mean_microwire, organized_data_microwire, trial_time, feature_dict, test_subject,
                                 test_session, event_lock, standardized_data,
                                 regularization_setting=regularization_setting,
                                 feature_names=[feature], data_modality=f'{electrode_selection} broadband LFP')

electrode_selection = 'macrocontact'
epochs_dataset, trial_time, microwire_names, feature_values = lfp_prep(test_subject, test_session,
                                                                       task, event_lock=event_lock,
                                                                       feature=feature,
                                                                       baseline=baseline, smooth=True,
                                                                       electrode_selection=electrode_selection,
                                                                       car=car_setting)
organized_data_mean_macro, organized_data_macro, feedback_dict = organize_data(epochs_dataset, feature_values,
                                                                   standardized_data=standardized_data,
                                                                   method='dPCA')

feature_dict = feedback_dict

# suptitle=f'All microwires, bandpassed at {bp}, {lock}-locked'
# plot_signal_avg(organized_data_mean, test_subject, test_session, trial_time, labels=feedback_dict,
#                     extra_string=f'Normalization = {standardized_data} {event_lock}-lock',
#                 signal_names=microwire_names)
dpca_3, Z_3 = dpca_plot_analysis(organized_data_mean_macro, organized_data_macro, trial_time, feature_dict, test_subject,
                                 test_session, event_lock, standardized_data,
                                 regularization_setting=regularization_setting,
                                 feature_names=[feature], data_modality=f'{electrode_selection} broadband LFP')