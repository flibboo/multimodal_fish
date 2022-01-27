import numpy as np
import pandas as pd

def flatten_fish(fish_name, fish_array):
    curr_fish = fish_array[fish_name]
    curr_fish = curr_fish.dropna()
    curr_fish_array = np.array(curr_fish)
    curr_fish_flattened_array = np.concatenate(curr_fish_array).ravel()
    return curr_fish_flattened_array

def fish_regression(fish_name, fish_array):
    pass

def correct_trials_per_day(fish_array):
    pass
    #return(percentages)

def plot_scatter():
    pass