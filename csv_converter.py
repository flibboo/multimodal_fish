import matplotlib.pyplot as plt
import numpy as np
from itertools import repeat
from IPython import embed
import statistics
from scipy.optimize import curve_fit
import os as os
from sklearn.linear_model import LogisticRegression
import pandas as pd
import openpyxl


# This portion of the code opens the xls file and creates a Pandas array with each fish getting a column and each row representing a day. 
# From each column, we then extract a numpy array, flatten it, and get a big array with all the trials for one fish over all the days. 
# This is then used in the logistic regression

curr_filepath = os.getcwd()
fig_filepath_base = os.path.dirname(curr_filepath)
fig_filepath = os.path.join(fig_filepath_base, 'figures')
if not os.path.exists(fig_filepath):
    os.mkdir(fig_filepath)

# excel to pandas

training = "Training_Messreihe.xlsx"
# df_1 = pd.read_excel(training, sheet_name="25.08.2021", engine='openpyxl')
# df_2 = pd.read_excel(training, sheet_name=[1], engine='openpyxl')
df = pd.read_excel(training, sheet_name=None, engine='openpyxl')

keys = df.keys() # keys = sheetname(dates)
all_fish = []

for key in keys: # geht durch die sheets
    df_run = df[key]
    column_names = df_run.columns
    for name in column_names:
        if name not in all_fish:
            if name.startswith('2020'):
                all_fish.append(name)


fish_dataframe = pd.DataFrame(index=np.arange(len(keys)), columns=all_fish)
low_fish_dataframe = pd.DataFrame(index=np.arange(len(keys)), columns=all_fish)
high_fish_dataframe = pd.DataFrame(index=np.arange(len(keys)), columns=all_fish)


# all fish, all stimuli
for index, key in enumerate(keys):
    df_run = df[key]
    for fish in all_fish:
        if fish in df_run.columns:
            curr_data = df_run[fish]
            curr_data = curr_data.dropna()
            fish_dataframe.at[index, fish] = curr_data

fish_dataframe.to_hdf("training_dataframe.hf", key="df")

# all fish, stimulus sorted
for index, key in enumerate(keys):
    df_run = df[key]
    for fish in all_fish:
        if fish in df_run.columns:

            curr_data = df_run[fish]
            stim_data = df_run["Stimulus"]

            stim_high_index = stim_data[stim_data == 'high'].index

            if len(stim_high_index) == 0:
                continue
            stim_high_data = curr_data[stim_high_index]
            stim_high_data = stim_high_data.dropna()
            high_fish_dataframe.at[index, fish] = stim_high_data

for index, key in enumerate(keys):
    df_run = df[key]
    for fish in all_fish:
        if fish in df_run.columns:

            curr_data = df_run[fish]
            stim_data = df_run["Stimulus"]

            stim_low_index = stim_data[stim_data == 'low'].index
            if len(stim_low_index) == 0:
                continue
            stim_low_data = curr_data[stim_low_index]
            stim_low_data = stim_low_data.dropna()
            low_fish_dataframe.at[index, fish] = stim_low_data


low_fish_dataframe.dropna()
low_fish_dataframe.to_hdf("training_low_dataframe.hf", key="df")
high_fish_dataframe.dropna()
high_fish_dataframe.to_hdf("training_high_dataframe.hf", key="df")

# creating the testing table
testing = "Testing.xlsx"
df = pd.read_excel(testing, sheet_name=None, engine='openpyxl')
keys = df.keys() # keys = sheetname(dates)
all_fish = []

for key in keys: # geht durch die sheets
    df_run = df[key]
    column_names = df_run.columns
    for name in column_names:
        if name not in all_fish:
            if name.startswith('2020'):
                all_fish.append(name)

"""
testing - dataframe creation
"""
# all stimuli - testing
testing_dataframe = pd.DataFrame(index=np.arange(len(keys)), columns=all_fish)

for index, key in enumerate(keys):
    df_run = df[key]
    for fish in all_fish:
        if fish in df_run.columns:
            curr_data = df_run[fish]
            curr_data = curr_data.dropna()
            testing_dataframe.at[index, fish] = curr_data

testing_dataframe.to_hdf("testing_dataframe.hf", key="df")

# low stimuli - testing
low_testing_dataframe = pd.DataFrame(index=np.arange(len(keys)), columns=all_fish)

for index, key in enumerate(keys):
    df_run = df[key]
    for fish in all_fish:
        if fish in df_run.columns:

            curr_data = df_run[fish]
            stim_data = df_run["Stimulus"]

            stim_low_index = stim_data[stim_data == 'low'].index
            if len(stim_low_index) == 0:
                continue
            stim_low_data = curr_data[stim_low_index]
            stim_low_data = stim_low_data.dropna()
            low_testing_dataframe.at[index, fish] = stim_low_data

low_testing_dataframe.to_hdf("testing_low_dataframe.hf", key="df")

# high stimuli - testing
high_testing_dataframe = pd.DataFrame(index=np.arange(len(keys)), columns=all_fish)

for index, key in enumerate(keys):
    df_run = df[key]
    for fish in all_fish:
        if fish in df_run.columns:

            curr_data = df_run[fish]
            stim_data = df_run["Stimulus"]

            stim_high_index = stim_data[stim_data == 'high'].index
            if len(stim_high_index) == 0:
                continue
            stim_high_data = curr_data[stim_high_index]
            stim_high_data = stim_high_data.dropna()
            high_testing_dataframe.at[index, fish] = stim_high_data

high_testing_dataframe.to_hdf("testing_high_dataframe.hf", key="df")

# mixed stimuli - testing
mixed_testing_dataframe = pd.DataFrame(index=np.arange(len(keys)), columns=all_fish)

for index, key in enumerate(keys):
    df_run = df[key]
    for fish in all_fish:
        if fish in df_run.columns:

            curr_data = df_run[fish]
            stim_data = df_run["Stimulus"]

            stim_mixed_index = stim_data[stim_data == 'mixed'].index
            if len(stim_mixed_index) == 0:
                continue
            stim_mixed_data = curr_data[stim_mixed_index]
            stim_mixed_data = stim_mixed_data.dropna()
            mixed_testing_dataframe.at[index, fish] = stim_mixed_data

mixed_testing_dataframe.to_hdf("testing_mixed_dataframe.hf", key="df")


"""
reaction time - dataframe creation
"""
testing_reaction_time = "Testing_reaction_time.xlsx"
df = pd.read_excel(testing_reaction_time, sheet_name=None, engine='openpyxl')
keys = df.keys() # keys = sheetname(dates)
testing_reaction_time_dataframe = pd.DataFrame(index=np.arange(len(keys)), columns=all_fish)

for index, key in enumerate(keys):
    df_run = df[key]
    for fish in all_fish:
        if fish in df_run.columns:
            curr_data = df_run[fish]
            curr_data = curr_data.dropna()
            testing_reaction_time_dataframe.at[index, fish] = curr_data

testing_reaction_time_dataframe.to_hdf("testing_reaction_time_dataframe.hf", key="df")

correct_reactions_dataframe = pd.DataFrame(index=np.arange(len(keys)), columns=all_fish) # create empty frame

for fish in testing_dataframe.columns:
    curr_fish_reaction = testing_reaction_time_dataframe[fish] # data for the times
    curr_fish_data = testing_dataframe[fish] # data for the choices
    for index, day in enumerate(curr_fish_data):
        correct_trials = []
        curr_fish_curr_day = curr_fish_data[index]
        curr_fish_curr_react = curr_fish_reaction[index]
        if len(curr_fish_curr_react) == 0: # skip the first day(no data)
            continue
        for trial_num, trial in enumerate(curr_fish_curr_day):
            if trial == 1: # only takes the right ones
                correct_trials.append(curr_fish_curr_react[trial_num])
        correct_reactions_dataframe.at[index, fish] = correct_trials

correct_reactions_dataframe.to_hdf("correct_testing_react_time_dataframe.hf", key="df")

testing_stim_frame = pd.DataFrame(index=np.arange(len(keys)), columns=all_fish)
for fish in all_fish: # to create a column für every fish, but its always the same for each fish anyway
    for index, key in enumerate(keys):
        df_run = df[key]
        curr_data = df_run["Stimulus"]
        testing_stim_frame.at[index, fish] = curr_data

testing_stim_frame.to_hdf("testing_stim_frame.hf", key="df")