from time import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import statistics
from scipy import stats
import scipy as sc
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
        for index, date in enumerate(curr_fish_data):
            curr_date_data = curr_fish_data.iloc[index]
            if str(curr_date_data) != 'nan':
                curr_date_data = curr_date_data.dropna()
                curr_date_data_len = len(curr_date_data)
                l.append(np.round((np.sum(curr_date_data) / curr_date_data_len), 3))
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
        time_list = list(range(1, (time + 1)))

        ax.plot(time_list, curr_data, "lightgrey", linewidth=0.8)
        x = time_list
        N = time

        # linear regression (handmade)
        y = curr_data
        E_xy = sum([a * b for a, b in zip(y, x)])
        E_x = sum(x)
        E_y = sum(y)
        E_x_2 = sum([a * b for a, b in zip(x, x)])

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
    E_y_med = np.median(E_y_sum)
    E_x_med = statistics.median(E_x_sum)
    E_xy_med = statistics.median(E_xy_sum)
    E_x_2_med = statistics.median(E_x_2_sum)
    m_median = statistics.median(m_summed)
    b_median = statistics.median(b_summed)

    # m_sum = (((N * E_xy_med) - (E_x_med * E_y_med)) / ((N * E_x_2_med) - (E_x_med * E_x_med)))
    # b_sum = (E_y_med - (m_sum * E_x_med)) / N

    line_calc = [m_median * x_l + b_median for x_l in time_list]  # loop is for multiplying lists

    ax.plot(time_list, line_calc, "black", linewidth=2)

    ax.set_xlabel('days')
    ax.set_ylabel('correct choices in %')
    ax.set_xlim([0, (time + 1)])
    ax.set_ylim([0, 1.05])
    plt.title("%s" % plot_name)

    return plt


def plot_single(percentages, all_fish, plot_name_single, tag):
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
        ax.plot(x, m_1 * time_array + b_1, linewidth=2)
        #print("%s slope:" % fish, m_1)

        # Pearson for R & p-Value
        r, p = np.round(sc.stats.pearsonr(x,y), 4)
        ax.text(30, 1.01, "R: %s  p: %s" % (r, p))
        
        # 50% line
        x_more = [0] + time_list + [len(curr_data)+1]
        ax.plot(x_more, ([0.5]*len(x_more)), linewidth= 0.5, linestyle='--', color="grey")

        if tag == "use vertical lines":
            # if statements for different training days
            if fish == "2020albi01": fish_num = 15
            if fish == "2020albi02": fish_num = 19
            if fish == "2020albi03": fish_num = 20
            if fish == "2020albi04": fish_num = 16
            if fish == "2020albi05": fish_num = 19
            if fish == "2020albi06": fish_num = 16

            x = 7  # first training days
            plt.axvline(x, color="lightgrey", linestyle=':')  # first days mixed (7 days)
            plt.axvline(x=(fish_num + x), color="lightgrey", linestyle=':')  # only high training (equals fish_num)
            plt.axvline(x=(fish_num + x + 17), color="lightgrey", linestyle=':')  # only low training (17 days)
            plt.axvline(x=(fish_num + x + 17 + 3), color="lightgrey", linestyle=':')  # low + high training (3 days)

        ax.set_xlabel('days')
        ax.set_ylabel('correct choices in %')
        ax.set_xlim([0, (time + 1)])
        ax.set_ylim([0, 1.05])
        plt.title("%s %s" % (fish, plot_name_single))

    return plt


def low_data_use(training_low_data, all_fish, plot_name, plot_name_single):
    percentages = percentage_creation(training_low_data)

    plot_all_together(percentages, all_fish, plot_name)
    #plt.show()
    plt.close()

    tag = "dont use vertical lines"  # this tag is for filtering out a graphic add, which is sensless here
    plot_single(percentages, all_fish, plot_name_single, tag)
    #plt.show()
    plt.close()

    return percentages


def high_data_use(training_high_data, all_fish, plot_name, plot_name_single):
    percentages = percentage_creation(training_high_data)
    plot_all_together(percentages, all_fish, plot_name)
    # plt.show()
    plt.close()

    tag = "dont use vertical lines"  # this tag is for filtering out a graphic add, which is sensless here
    plot_single(percentages, all_fish, plot_name_single, tag)
    # plt.show()
    plt.close()

    return percentages


def fish_regression(fish, flattened_fish, percentages, plot_name_single):
    time = len(flattened_fish)
    for percentage, name in zip(percentages, fish):
        x_axis = np.arange(time).reshape(-1, 1)

        bool_trial = flattened_fish > 0.7
        y_axis = bool_trial * 1

        # y_axis = percentages["perc_%s" % fish]
        # y_axis = flattened_fish
        # model = LogisticRegression(solver='liblinear', random_state=0)
        model = LogisticRegression(C=10.0, class_weight=None, dual=False, fit_intercept=True,
                                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                                   multi_class='ovr', n_jobs=None, penalty='l2',
                                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                                   warm_start=False)
        model.fit(x_axis, y_axis)
        model.predict_proba(x_axis)  # shows performance of the model
        model.predict(x_axis)  # shows the predictions
        # print(model.score(x_axis, y_axis))  # shows the accuracy
        plt.scatter(x_axis, y_axis)
        plt.title("%s %s" % (fish, plot_name_single))
        plt.plot(x_axis, model.predict_proba(x_axis)[:, 1])

    return plt


def diverse_statistics(percentages, flattened_fish, data_mixed, data_high, data_low):
    # those lists need some adjustment, because of the different time periods
    first_day = []
    last_day = []

    time = len(flattened_fish)
    time_list = list(range(1, (time + 1)))
    for percentage in percentages:
        curr_perc = np.array(percentages["%s" % percentage])
        # print(stats.shapiro(curr_perc))  # Shapiro-Wilk-Test
        # print(np.corrcoef(time_list, curr_perc))  # Pearson Correlation
        # print(stats.spearmanr(time_list, curr_perc, axis=1))  # Spearman Correlation TIME CORRECT?! want int instead of list
        # first_day.append(curr_perc[0])
        # last_day.append(curr_perc[-1])

    #print(stats.ttest_rel(first_day, last_day))

    # comparison between the different median of the fish for each stim
    high_test_perc = percentage_creation(data_high)
    low_test_perc = percentage_creation(data_low)
    mixed_test_perc = percentage_creation(data_mixed)

    for fish in high_test_perc:
        print("High median of %s:" % fish, np.median(high_test_perc[fish]))

    for fish in low_test_perc:
        print("Low median of %s:" % fish, np.median(low_test_perc[fish]))

    for fish in mixed_test_perc:
        print("Mixed median of %s:" % fish, np.median(mixed_test_perc[fish]))

    return percentages


def boxplotting(data_high, data_low, data_mixed):
    high_test_perc = percentage_creation(data_high)
    low_test_perc = percentage_creation(data_low)
    mixed_test_perc = percentage_creation(data_mixed)

    yr_highness = []
    for fish in high_test_perc:
        if fish == "perc_2020albi05" or fish == "perc_2020albi06": # can be skipped, if all fish should be included
            yr_highness.extend(high_test_perc[fish])


    yr_lowness = []
    for fish in low_test_perc:
        if fish == "perc_2020albi05" or fish == "perc_2020albi06":
            yr_lowness.extend(low_test_perc[fish])

    yr_mixedness = []
    for fish in mixed_test_perc:
        if fish == "perc_2020albi05" or fish == "perc_2020albi06":
            yr_mixedness.extend(mixed_test_perc[fish])

    #yr_mixedness = np.array(yr_mixedness)
    #yr_mixedness = yr_mixedness[yr_mixedness > 0.3]

    data = [yr_highness, yr_lowness, yr_mixedness]
    fig, ax = plt.subplots()
    ax.set_title('comparison of the testing stimuli')
    ax.boxplot(data)

    plt.xticks([1, 2, 3], ['high', 'low', 'mixed'])
    ax.set_ylabel('correct choices in %')
    plt.show()
    #plt.savefig("comparison of the testing stimuli.png")
    sc.stats.ttest_ind(yr_highness, yr_mixedness)

    return plt


def reaction_time_analysis(times, data, stim):
    # each frame contains a needed information: the data, the reaction times and the used stimuli
    high_react_ls = []
    low_react_ls = []
    mixed_react_ls = []

    for fish in data.columns:
        if fish == "2020albi05" or fish == "2020albi06": # can be skipped, if all fish should be included
            for index in data.index:
                if index == 0:  # for skipping the first testing day (no time data)
                    continue
                    # getting the data for one fish
                curr_data = data[fish][index]
                curr_times = times[fish][index]
                curr_stim = stim[fish][index]

                arr_data = np.array(curr_data)
                arr_times = np.array(curr_times)
                arr_stim = np.array(curr_stim)

                # high stim
                high_data = arr_data[arr_stim == "high"]  # only the choices where the stim was high
                high_times = arr_times[arr_stim == "high"]
                high_stim = arr_stim[arr_stim == "high"]

                right_high_times = high_times[high_data == 1]  # only the times where the fish choices where correct
                high_react_ls.extend(right_high_times)

                # low stim
                low_data = arr_data[arr_stim == "low"]
                low_times = arr_times[arr_stim == "low"]
                low_stim = arr_stim[arr_stim == "low"]

                right_low_times = low_times[low_data == 1]
                low_react_ls.extend(right_low_times)

                # mixed stim
                mixed_data = arr_data[arr_stim == "mixed"]
                mixed_times = arr_times[arr_stim == "mixed"]
                mixed_stim = arr_stim[arr_stim == "mixed"]

                right_mixed_times = mixed_times[mixed_data == 1]
                mixed_react_ls.extend(right_mixed_times)

    data = [high_react_ls, low_react_ls, mixed_react_ls]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title('comparison of the testing stimuli in reaction time', fontsize=13)
    ax.boxplot(data)
    plt.xticks([1, 2, 3], ['high', 'low', 'mixed'], fontsize=12)
    ax.set_ylabel('reaction time [s]', fontsize=12)

    plt.savefig("comparison of the testing stimuli in reaction time.svg")

    return plt
