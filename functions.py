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


def plot_all_together(percentages, all_fish, plot_name):
    fig, ax = plt.subplots()

    E_xy_sum = []
    E_x_sum = []
    E_y_sum = []
    E_x_2_sum = []
    m_summed = []
    b_summed = []

    for fish in all_fish:
        curr_data = percentages["perc_%s" % fish]
        time = len(curr_data)
        time_array = np.array(time)
        time_list = list(range(1, (time+1)))

        ax.plot(time_list, curr_data, "lightgrey", linewidth=0.8)
        x = time_list
        N = time

        # linear regression (handmade)
        y = curr_data
        E_xy = sum([a * b for a, b in zip(y, x)])
        E_x = sum(x)
        E_y = sum(y)
        E_x_2 = sum([a*b for a, b in zip(x, x)])

        m = (((N * E_xy) - (E_x * E_y)) / ((N * E_x_2) - (E_x * E_x)))
        b = (E_y - (m * E_x)) / N

        line_calc = [m * x_l + b for x_l in time_list]

        ax.plot(time_list, line_calc, "lightgrey", linewidth=0.8)
        # print('y =', m, 'x +', b)

        # summed axes
        m_summed.append(m)
        b_summed.append(b)
        E_x_sum.append(E_x)
        E_y_sum.append(E_y)
        E_xy_sum.append(E_xy)
        E_x_2_sum.append(E_x_2)

    # summed regression
    E_y_med = statistics.median(E_y_sum)
    E_x_med = statistics.median(E_x_sum)
    E_xy_med = statistics.median(E_xy_sum)
    E_x_2_med = statistics.median(E_x_2_sum)
    m_median = statistics.median(m_summed)
    b_median = statistics.median(b_summed)

    m_sum = (((N * E_xy_med) - (E_x_med * E_y_med)) / ((N * E_x_2_med) - (E_x_med * E_x_med)))
    b_sum = (E_y_med - (m_sum * E_x_med)) / N

    line_calc = [m_median * x_l + b_median for x_l in time_list] # loop is for multiplying lists
    ax.plot(time_list, line_calc, "black", linewidth=2)


    ax.set_xlabel('days')
    ax.set_ylabel('correct choices in %')
    ax.set_xlim([0, (time+1)])
    ax.set_ylim([0, 1.05])
    plt.title("%s" % plot_name)

    return ax


def plot_single(percentages, all_fish, plot_name_single):

    for fish in all_fish:
        curr_data = percentages["perc_%s" % fish]
        time = len(curr_data)
        time_list = list(range(1, (time + 1)))
        time_array = np.array(time_list)

        fig, ax = plt.subplots()
        ax.plot(time_list, curr_data)
        y = curr_data
        x = time_list
        N = time
        """
        brauche ich grade nicht(aber falls doch bleibts hier)
        
        # linear regression (handmade)
        
        E_xy = sum([a * b for a, b in zip(y, x)])
        E_x = sum(x)
        E_y = sum(y)
        E_x_2 = sum([a * b for a, b in zip(x, x)])

        m = (((N * E_xy) - (E_x * E_y)) / ((N * E_x_2) - (E_x * E_x)))
        b = (E_y - (m * E_x)) / N

        line_calc = [m * x_l + b for x_l in time_list]
        ax.plot(time_list, line_calc, "black", linewidth=0.8)
        # print('y =', m, 'x +', b)
        """
        m_1, b_1 = np.polyfit(x, y, 1)
        ax.plot(x, m_1*time_array + b_1, linewidth=2)


        ax.set_xlabel('days')
        ax.set_ylabel('correct choices in %')
        ax.set_xlim([0, (time + 1)])
        ax.set_ylim([0, 1.05])
        plt.title("%s %s" % (fish, plot_name_single))

    return ax


def low_data_use(training_low_data, all_fish, plot_name, plot_name_single):
    percentages = percentage_creation(training_low_data)
    plot_all_together(percentages, all_fish, plot_name)
    plt.show()

    plot_single(percentages, all_fish, plot_name_single)
    plt.show()

    return percentages, plot_all_together, plot_single


def high_data_use(training_high_data, all_fish, plot_name, plot_name_single):
    percentages = percentage_creation(training_high_data)
    plot_all_together(percentages, all_fish, plot_name)
    plt.show()
    plot_single(percentages, all_fish, plot_name_single)
    plt.show()

    return percentages, plot_all_together, plot_single


def fish_regression(fish, flattened_fish, percentages, plot_name_single):

    time = len(flattened_fish)
    for percentage, name in zip(percentages, fish):
        x_axis = np.arange(time).reshape(-1, 1)

        bool_trial = flattened_fish > 0.7
        y_axis = bool_trial * 1

        #y_axis = percentages["perc_%s" % fish]
        #y_axis = flattened_fish
        # model = LogisticRegression(solver='liblinear', random_state=0)
        model = LogisticRegression(C=10.0, class_weight=None, dual=False, fit_intercept=True,
                                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                                   multi_class='ovr', n_jobs=None, penalty='l2',
                                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                                   warm_start=False)
        model.fit(x_axis, y_axis)
        model.predict_proba(x_axis)  # shows performance of the model
        model.predict(x_axis)  # shows the predictions
        #print(model.score(x_axis, y_axis))  # shows the accuracy
        plt.scatter(x_axis, y_axis)
        plt.title("%s %s" % (fish, plot_name_single))
        plt.plot(x_axis, model.predict_proba(x_axis)[:, 1])

    return plt

