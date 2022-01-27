import numpy as np
import pandas as pd
from IPython import embed

def flatten_fish(fish_name, fish_array):
    curr_fish = fish_array[fish_name]
    curr_fish = curr_fish.dropna()
    curr_fish_array = np.array(curr_fish)
    curr_fish_flattened_array = np.concatenate(curr_fish_array).ravel()
    return curr_fish_flattened_array

def percentage_creation(dataframe):
    count = 0
    for name in dataframe.columns:
        df = dataframe[name][count]
        df = df.dropna()
        print(first_column=df.loc[:, 0])
        embed()
        quit()
    pass

def fish_regression(fish_name, fish_array):
    for percentage, name in zip(all_percentages, names):
        x_axis = np.arange(len(time)).reshape(-1, 1)

        bool_trial = percentage > 0.7
        y_axis = bool_trial * 1

        # model = LogisticRegression(solver='liblinear', random_state=0)
        model = LogisticRegression(C=10.0, class_weight=None, dual=False, fit_intercept=True,
                                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                                   multi_class='ovr', n_jobs=None, penalty='l2',
                                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                                   warm_start=False)
        model.fit(x_axis, y_axis)
        model.predict_proba(x_axis)  # shows performance of the model
        model.predict(x_axis)  # shows the predictions
        print(model.score(x_axis, y_axis))  # shows the accuracy
    pass

def correct_trials_per_day(fish_array):
    pass
    #return(percentages)

def plot_scatter():
    pass