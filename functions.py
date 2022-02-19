import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import statistics
from IPython import embed


def flatten_fish(fish_name, fish_array):
    curr_fish = fish_array[fish_name]
    curr_fish = curr_fish.dropna()
    curr_fish_array = np.array(curr_fish)
    curr_fish_flattened_array = np.concatenate(curr_fish_array).ravel()

    return curr_fish_flattened_array


def percentage_creation(dataframe):
    percentages = {}

    for name in dataframe.columns:
        l = []
        dkey = 'perc_%s' % name
        curr_fish_data = dataframe[name]
        for index,date in enumerate(curr_fish_data):
            curr_date_data = curr_fish_data.iloc[index]
            if str(curr_date_data) != 'nan':
                curr_date_data = curr_date_data.dropna()
                curr_date_data_len = len(curr_date_data)
                l.append(np.round((np.sum(curr_date_data) / curr_date_data_len),3))
        percentages.update({dkey: l})
    return percentages


def fish_regression(fish, flattened_fish, percentages):

    time = len(flattened_fish)  # wie bekomme ich die Zeit?
    print(time)
    embed()
    quit()
    for percentage, name in zip(percentages, fish):
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
        plt.scatter(x_axis, y_axis)
        plt.plot(x_axis, model.predict_proba(x_axis)[:, 1])

    return plt


def plot_all_together(percentages, time):
    # all plots in one graphic
    fig, ax = plt.subplots()
    time_array = np.array(time)

    E_xy_sum = []
    E_x_sum = []
    E_y_sum = []
    E_x_2_sum = []

    for percentage in percentages:
        ax.plot(time, percentage, "lightgrey", linewidth=0.8)
        x = time_array
        N = 23  # wie zähle ich die sheets????

        # linear regression (handmade)
        y = percentage
        E_xy = sum(time_array * percentage)
        E_x = sum(time_array)
        E_y = sum(percentage)
        E_x_2 = sum(x * x)
        m = (((N * E_xy) - (E_x * E_y)) / ((N * E_x_2) - (E_x * E_x)))
        b = (E_y - (m * E_x)) / N
        ax.plot(time, m * time_array + b, linewidth=0.8)
        # print('y =', m, 'x +', b)

        # summed axes
        E_x_sum.append(E_x)
        E_y_sum.append(E_y)
        E_xy_sum.append(E_xy)
        E_x_2_sum.append(E_x_2)

    # summed regression
    E_y_med = statistics.median(E_y_sum)
    E_x_med = statistics.median(E_x_sum)
    E_xy_med = statistics.median(E_xy_sum)
    E_x_2_med = statistics.median(E_x_2_sum)

    m_sum = (((N * E_xy_med) - (E_x_med * E_y_med)) / ((N * E_x_2_med) - (E_x_med * E_x_med)))
    b_sum = (E_y_med - (m * E_x_med)) / N
    ax.plot(time, m_sum * time_array + b_sum, "black", linewidth=2)

    # ax.plot(range(0, 24), threshold, '--', linewidth=0.8)  # 80% Grenze
    # ax.plot(range(0, 24), midline, '--', linewidth=0.8)  # 50% Grenze

    ax.set_xlabel('days')
    ax.set_ylabel('correct choices in %')
    ax.set_xlim([0, 24])
    ax.set_ylim([0, 1.05])

    return ax

def plot_single(percentages, time):
    # one plot per fish

    time_array = np.array(time)

    E_xy_sum = []
    E_x_sum = []
    E_y_sum = []
    E_x_2_sum = []

    for percentage in percentages:
        fig, ax = plt.subplots()
        ax.plot(time, percentage, "lightgrey", linewidth=0.8)
        x = time_array
        N = 23  # wie zähle ich die sheets????

        # linear regression (handmade)
        y = percentage
        E_xy = sum(time_array * percentage)
        E_x = sum(time_array)
        E_y = sum(percentage)
        E_x_2 = sum(x * x)
        m = (((N * E_xy) - (E_x * E_y)) / ((N * E_x_2) - (E_x * E_x)))
        b = (E_y - (m * E_x)) / N
        ax.plot(time, m * time_array + b, linewidth=0.8)
        # print('y =', m, 'x +', b)

        # summed axes
        E_x_sum.append(E_x)
        E_y_sum.append(E_y)
        E_xy_sum.append(E_xy)
        E_x_2_sum.append(E_x_2)

    # summed regression
    E_y_med = statistics.median(E_y_sum)
    E_x_med = statistics.median(E_x_sum)
    E_xy_med = statistics.median(E_xy_sum)
    E_x_2_med = statistics.median(E_x_2_sum)

    m_sum = (((N * E_xy_med) - (E_x_med * E_y_med)) / ((N * E_x_2_med) - (E_x_med * E_x_med)))
    b_sum = (E_y_med - (m * E_x_med)) / N
    ax.plot(time, m_sum * time_array + b_sum, "black", linewidth=2)

    # ax.plot(range(0, 24), threshold, '--', linewidth=0.8)  # 80% Grenze
    # ax.plot(range(0, 24), midline, '--', linewidth=0.8)  # 50% Grenze

    ax.set_xlabel('days')
    ax.set_ylabel('correct choices in %')
    ax.set_xlim([0, 24])
    ax.set_ylim([0, 1.05])

    return ax