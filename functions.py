from time import time
from turtle import width
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

def binomial_data(low_frame, high_frame):
    bino_low_frame = {}
    bino_high_frame = {}

    for name in low_frame.columns:
        temp_correct = []
        temp_trials = []
        dkey = 'bino_%s' % name
        curr_fish_data = low_frame[name]
        for index, date in enumerate(curr_fish_data):
            curr_date_data = curr_fish_data.iloc[index]
            if str(curr_date_data) != 'nan':
                curr_date_data = curr_date_data.dropna()
                curr_date_data_len = len(curr_date_data)
                temp_correct.append(np.sum(curr_date_data))
                temp_trials.append(curr_date_data_len)
        temp_c = np.add.reduceat(temp_correct, np.arange(0, len(temp_correct), 3)) # gets tuple of 3 and sums them
        temp_t = np.add.reduceat(temp_trials, np.arange(0, len(temp_trials), 3))
        bino_low_list = []
        for index, thirds in enumerate(temp_c): # makes a binomal test for every 3 days
            bino_low_list.append(sc.stats.binom_test(x=temp_c[index], n=temp_t[index], p=0.5))
            bino_low_frame.update({dkey: bino_low_list})

    for name in high_frame.columns:
        temp_correct = []
        temp_trials = []
        dkey = 'bino_%s' % name
        curr_fish_data = high_frame[name]
        for index, date in enumerate(curr_fish_data):
            curr_date_data = curr_fish_data.iloc[index]
            if str(curr_date_data) != 'nan':
                curr_date_data = curr_date_data.dropna()
                curr_date_data_len = len(curr_date_data)
                temp_correct.append(np.sum(curr_date_data))
                temp_trials.append(curr_date_data_len)
        temp_c = np.add.reduceat(temp_correct, np.arange(0, len(temp_correct), 3)) # gets tuple of 3 and sums them
        temp_t = np.add.reduceat(temp_trials, np.arange(0, len(temp_trials), 3))
        bino_high_list = []
        for index, thirds in enumerate(temp_c): # makes a binomal test for every 3 days
            bino_high_list.append(sc.stats.binom_test(x=temp_c[index], n=temp_t[index], p=0.5))
            bino_high_frame.update({dkey: bino_high_list})

    return bino_low_frame, bino_high_frame

def plot_all_together(percentages, all_fish, plot_name):
    fig, ax = plt.subplots(figsize=(10,8))

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

        ax.plot(time_list, curr_data, "lightgrey", linewidth=0.6)
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

        ax.plot(time_list, line_calc, "powderblue", linewidth=0.7)

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
    print('y =', m_median, 'x +', b_median)
    # m_sum = (((N * E_xy_med) - (E_x_med * E_y_med)) / ((N * E_x_2_med) - (E_x_med * E_x_med)))
    # b_sum = (E_y_med - (m_sum * E_x_med)) / N

    line_calc = [m_median * x_l + b_median for x_l in time_list]  # loop is for multiplying lists

    ax.plot(time_list, line_calc, "darkviolet", linewidth=2)

    ax.set_xlabel('Tage')
    ax.set_ylabel('richtige Entscheidungen in %')
    ax.set_xlim([0, (time + 1)])
    ax.set_ylim([0, 1.05])
    ax.yaxis.set_ticks(np.arange(0, 1.05, step=0.1))
    plt.title("%s" % plot_name)
    #plt.savefig("/home/efish/PycharmProjects/philipp/figures/%s.svg" %plot_name)
    #plt.savefig("/home/efish/PycharmProjects/philipp/figures/%s.png" %plot_name)

    return plt


def plot_single(percentages, all_fish, plot_name_single, tag, binomial_dataframe_low, binomial_dataframe_high):
    # universal plot variables
    y_ticks_single = 1.05
    if tag == "use vertical lines":
        y_lims_single = 1.05
    else:    
        y_lims_single = 1.19


    for fish in all_fish:
        curr_data = percentages["perc_%s" % fish]
        time = len(curr_data)
        time_list = list(range(1, (time + 1)))
        time_array = np.array(time_list)

        fig, ax = plt.subplots(1 ,2, figsize=(10,8), gridspec_kw={'width_ratios': [4, 1]})

        ax[0].plot(time_list, curr_data)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)

        y = curr_data
        x = time_list
        N = time

        # lineare Regression
        m_1, b_1 = np.polyfit(x, y, 1)
        ax[0].plot(x, m_1 * time_array + b_1, linewidth=1.5)
        m_1 = np.round(m_1, 3)
        b_1 = np.round(b_1, 3)
        print("%s: y = %s x + %s" % (fish, m_1, b_1))

        # 50% line
        ax[0].axhline(0.5, linewidth= 0.5, linestyle='--', color="grey")

        # Pearson for R & p-Value
        r, p = np.round(sc.stats.pearsonr(x,y), 4)
        ax[0].text(5, 0.1, "R: %s  p: %s" % (r, p), bbox=dict(boxstyle = "square", facecolor = "white"), ha='center', va='center')
        
        if tag == "use vertical lines":
            # if statements for different training days
            if fish == "2020albi01": fish_num = 15
            if fish == "2020albi02": fish_num = 19
            if fish == "2020albi03": fish_num = 20
            if fish == "2020albi04": fish_num = 16
            if fish == "2020albi05": fish_num = 19
            if fish == "2020albi06": fish_num = 16

            x = 7  # first training days
            ax[0].axvline(x, color="lightgrey", linestyle=':')  # first days mixed (7 days)
            ax[0].axvline(x=(fish_num + x), color="lightgrey", linestyle=':')  # only high training (equals fish_num)
            ax[0].axvline(x=(fish_num + x + 17), color="lightgrey", linestyle=':')  # only low training (17 days)
            ax[0].axvline(x=(fish_num + x + 17 + 3), color="lightgrey", linestyle=':')  # low + high training (3 days)

        ax[0].set_xlabel('Tage')
        ax[0].set_ylabel('richtige Entscheidungen in %')
        ax[0].set_xlim([0, (time + 1)])
        ax[0].set_ylim([0, y_lims_single])
        ax[0].yaxis.set_ticks(np.arange(0, y_ticks_single, step=0.1))
        #ax[0].set_title("%s %s" % (fish, plot_name_single))

        # binomial visualisation, but only for high and low data
        if tag == "low use, no vert lines": # just used the tag because its only with high/low function
            text_filler = np.round(binomial_dataframe_low["bino_%s" %fish][0], 3)
            ax[0].axvline(x = 3, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_low["bino_%s" %fish][0] > 0.01:
                ax[0].text(1.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_low["bino_%s" %fish][0] > 0.001:
                ax[0].text(1.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_low["bino_%s" %fish][0] < 0.001:
                ax[0].text(1.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(1.5, 1.1 ,text_filler, ha='center', va='center')

            text_filler = np.round(binomial_dataframe_low["bino_%s" %fish][1], 3)
            ax[0].axvline(x = 6, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_low["bino_%s" %fish][1] > 0.01:
                ax[0].text(4.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_low["bino_%s" %fish][1] > 0.001:
                ax[0].text(4.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_low["bino_%s" %fish][1] < 0.001:
                ax[0].text(4.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(4.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_low["bino_%s" %fish][2], 3)
            ax[0].axvline(x = 9, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_low["bino_%s" %fish][2] > 0.01:
                ax[0].text(7.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_low["bino_%s" %fish][2] > 0.001:
                ax[0].text(7.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_low["bino_%s" %fish][2] < 0.001:
                ax[0].text(7.05, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(7.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_low["bino_%s" %fish][3], 3)
            ax[0].axvline(x = 12, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_low["bino_%s" %fish][3] > 0.01:
                ax[0].text(10.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_low["bino_%s" %fish][3] > 0.001:
                ax[0].text(10.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_low["bino_%s" %fish][3] < 0.001:
                ax[0].text(10.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(10.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_low["bino_%s" %fish][4], 3)
            ax[0].axvline(x = 15, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_low["bino_%s" %fish][4] > 0.01:
                ax[0].text(13.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_low["bino_%s" %fish][4] > 0.001:
                ax[0].text(13.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_low["bino_%s" %fish][4] < 0.001:
                ax[0].text(13.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(13.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_low["bino_%s" %fish][5], 3)
            ax[0].axvline(x = 18, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_low["bino_%s" %fish][5] > 0.01:
                ax[0].text(16.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_low["bino_%s" %fish][5] > 0.001:
                ax[0].text(16.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_low["bino_%s" %fish][5] < 0.001:
                ax[0].text(16.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(16.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_low["bino_%s" %fish][6], 3)
            ax[0].axvline(x = 21, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_low["bino_%s" %fish][6] > 0.01:
                ax[0].text(19.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_low["bino_%s" %fish][6] > 0.001:
                ax[0].text(19.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_low["bino_%s" %fish][6] < 0.001:
                ax[0].text(19.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(19.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_low["bino_%s" %fish][7], 3)
            ax[0].axvline(x = 24, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_low["bino_%s" %fish][7] > 0.01:
                ax[0].text(22.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_low["bino_%s" %fish][7] > 0.001:
                ax[0].text(22.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_low["bino_%s" %fish][7] < 0.001:
                ax[0].text(22.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(22.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_low["bino_%s" %fish][8], 3)
            ax[0].axvline(x = 27, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_low["bino_%s" %fish][8] > 0.01:
                ax[0].text(25.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_low["bino_%s" %fish][8] > 0.001:
                ax[0].text(25.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_low["bino_%s" %fish][8] < 0.001:
                ax[0].text(25.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(25.5, 1.1 ,text_filler, ha='center', va='center')

            try: # needed because not every fish made the same amount of trials
                text_filler = np.round(binomial_dataframe_low["bino_%s" %fish][9], 3)
                ax[0].axvline(x = 30, ymin = 0.88, ymax = 0.9)
                if 0.05 > binomial_dataframe_low["bino_%s" %fish][9] > 0.01:
                    ax[0].text(28.5, 1.05, '*', ha='center', va='center')
                if 0.01 > binomial_dataframe_low["bino_%s" %fish][9] > 0.001:
                    ax[0].text(28.5, 1.05, '**', ha='center', va='center')
                if binomial_dataframe_low["bino_%s" %fish][9] < 0.001:
                    ax[0].text(28.5, 1.05, '***', ha='center', va='center')
                    text_filler = "<0.001"
                ax[0].text(28.5, 1.1 ,text_filler, ha='center', va='center')
            except:
                print("Hier könnte Ihre Werbung stehen")
            ax[0].text((time/2), 1.15, "p-Werte über je 3 Tage", ha='center', va='center')
            

        if tag == "high use, no vert lines": 

            text_filler = np.round(binomial_dataframe_high["bino_%s" %fish][0], 3)
            ax[0].axvline(x = 3, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_high["bino_%s" %fish][0] > 0.01:
                ax[0].text(1.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_high["bino_%s" %fish][0] > 0.001:
                ax[0].text(1.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_high["bino_%s" %fish][0] < 0.001:
                ax[0].text(1.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(1.5, 1.1 ,text_filler, ha='center', va='center')

            text_filler = np.round(binomial_dataframe_high["bino_%s" %fish][1], 3)
            ax[0].axvline(x = 6, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_high["bino_%s" %fish][1] > 0.01:
                ax[0].text(4.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_high["bino_%s" %fish][1] > 0.001:
                ax[0].text(4.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_high["bino_%s" %fish][1] < 0.001:
                ax[0].text(4.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(4.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_high["bino_%s" %fish][2], 3)
            ax[0].axvline(x = 9, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_high["bino_%s" %fish][2] > 0.01:
                ax[0].text(7.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_high["bino_%s" %fish][2] > 0.001:
                ax[0].text(7.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_high["bino_%s" %fish][2] < 0.001:
                ax[0].text(7.05, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(7.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_high["bino_%s" %fish][3], 3)
            ax[0].axvline(x = 12, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_high["bino_%s" %fish][3] > 0.01:
                ax[0].text(10.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_high["bino_%s" %fish][3] > 0.001:
                ax[0].text(10.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_high["bino_%s" %fish][3] < 0.001:
                ax[0].text(10.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(10.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_high["bino_%s" %fish][4], 3)
            ax[0].axvline(x = 15, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_high["bino_%s" %fish][4] > 0.01:
                ax[0].text(13.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_high["bino_%s" %fish][4] > 0.001:
                ax[0].text(13.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_high["bino_%s" %fish][4] < 0.001:
                ax[0].text(13.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(13.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_high["bino_%s" %fish][5], 3)
            ax[0].axvline(x = 18, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_high["bino_%s" %fish][5] > 0.01:
                ax[0].text(16.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_high["bino_%s" %fish][5] > 0.001:
                ax[0].text(16.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_high["bino_%s" %fish][5] < 0.001:
                ax[0].text(16.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(16.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_high["bino_%s" %fish][6], 3)
            ax[0].axvline(x = 21, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_high["bino_%s" %fish][6] > 0.01:
                ax[0].text(19.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_high["bino_%s" %fish][6] > 0.001:
                ax[0].text(19.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_high["bino_%s" %fish][6] < 0.001:
                ax[0].text(19.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(19.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_high["bino_%s" %fish][7], 3)
            ax[0].axvline(x = 24, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_high["bino_%s" %fish][7] > 0.01:
                ax[0].text(22.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_high["bino_%s" %fish][7] > 0.001:
                ax[0].text(22.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_high["bino_%s" %fish][7] < 0.001:
                ax[0].text(22.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(22.5, 1.1 ,text_filler, ha='center', va='center')
            
            text_filler = np.round(binomial_dataframe_high["bino_%s" %fish][8], 3)
            ax[0].axvline(x = 27, ymin = 0.88, ymax = 0.9)
            if 0.05 > binomial_dataframe_high["bino_%s" %fish][8] > 0.01:
                ax[0].text(25.5, 1.05, '*', ha='center', va='center')
            if 0.01 > binomial_dataframe_high["bino_%s" %fish][8] > 0.001:
                ax[0].text(25.5, 1.05, '**', ha='center', va='center')
            if binomial_dataframe_high["bino_%s" %fish][8] < 0.001:
                ax[0].text(25.5, 1.05, '***', ha='center', va='center')
                text_filler = "<0.001"
            ax[0].text(25.5, 1.1 ,text_filler, ha='center', va='center')

            try: # needed because not every fish made the same amount of trials
                text_filler = np.round(binomial_dataframe_high["bino_%s" %fish][9], 3)
                ax[0].axvline(x = 30, ymin = 0.88, ymax = 0.9)
                if 0.05 > binomial_dataframe_high["bino_%s" %fish][9] > 0.01:
                    ax[0].text(28.5, 1.05, '*', ha='center', va='center')
                if 0.01 > binomial_dataframe_high["bino_%s" %fish][9] > 0.001:
                    ax[0].text(28.5, 1.05, '**', ha='center', va='center')
                if binomial_dataframe_high["bino_%s" %fish][9] < 0.001:
                    ax[0].text(28.5, 1.05, '***', ha='center', va='center')
                    text_filler = "<0.001"
                ax[0].text(28.5, 1.1 ,text_filler, ha='center', va='center')
            except:
                print("Hier könnte Ihre Werbung stehen")
            ax[0].text((time/2), 1.15, "p-Werte über je 3 Tage", ha='center', va='center')

        # boxplot 

        ax[1].boxplot(curr_data)
        ax[1].get_xaxis().set_ticks([])
        #ax[1].yaxis.set_label_position("right")
        #ax[1].yaxis.tick_right()
        #ax[1].set_ylabel('correct choices in %')
        #ax[1].yaxis.set_ticks(np.arange(0, y_ticks_single, step=0.1))
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].set_ylim([0, y_lims_single])
        ax[1].set_facecolor('none')
        ax[1].axis('off')

        plt.subplots_adjust(wspace=-0.1, hspace=0.2)
        #plt.savefig("/home/efish/PycharmProjects/philipp/figures/%s %s.svg" % (fish, plot_name_single))
        #plt.savefig("/home/efish/PycharmProjects/philipp/figures/%s %s.png" % (fish, plot_name_single))

    return plt


def low_data_use(training_low_data, all_fish, plot_name, plot_name_single, binomial_dataframe_low, binomial_dataframe_high):
    percentages = percentage_creation(training_low_data)

    plot_all_together(percentages, all_fish, plot_name)
    plt.show()
    #plt.close()

    tag = "low use, no vert lines"  # this tag is for filtering out a graphic add, which is sensless here
    plot_single(percentages, all_fish, plot_name_single, tag, binomial_dataframe_low, binomial_dataframe_high)
    plt.show()
    #plt.close()

    return percentages


def high_data_use(training_high_data, all_fish, plot_name, plot_name_single, binomial_dataframe_low, binomial_dataframe_high):
    percentages = percentage_creation(training_high_data)
    plot_all_together(percentages, all_fish, plot_name)
    plt.show()
    #plt.close()

    tag = "high use, no vert lines"  # this tag is for filtering out a graphic add, which is sensless here
    plot_single(percentages, all_fish, plot_name_single, tag, binomial_dataframe_low, binomial_dataframe_high)
    plt.show()
    #plt.close()

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
    ax.axhline(0.5, linewidth= 0.5, linestyle='--', color="grey")
    ax.set_title('Vergleich der Stimuli im Testversuch')
    ax.boxplot(data)

    plt.xticks([1, 2, 3], ['hoch', 'niedrig', 'gemischt'])
    ax.set_ylabel('richtige Entscheidungen in %')
    #plt.savefig("/home/efish/PycharmProjects/philipp/figures/Vergleich_der_Stimuli_im_Testversuch.svg")
    #plt.savefig("/home/efish/PycharmProjects/philipp/figures/Vergleich_der_Stimuli_im_Testversuch.png")
    #plt.savefig("/home/efish/PycharmProjects/philipp/figures/Vergleich_der_Stimuli_im_Testversuch_nur_05_06.svg")

    #ttest analysis for each combination
    t_h_l = sc.stats.ttest_ind(yr_highness, yr_lowness)
    t_h_m = sc.stats.ttest_ind(yr_highness, yr_mixedness)
    t_l_m = sc.stats.ttest_ind(yr_lowness, yr_mixedness)
    print("T-test Ergebnisse für korrekte Entscheidungen: hoch + niedrig: %s, hoch + gemischt: %s, niedrig + gemischt: %s" %(t_h_l, t_h_m, t_l_m))

    return plt

def boxplotting_for_singles (dataframe, tag):
    # universal plot variables
    y_lims_single = 1.05
    y_ticks_single = 1.05

    curr_perc = percentage_creation(dataframe)
    fix, ax = plt.subplots()
    all_fish_ls = []
    for fish in curr_perc:
        print(fish)
        all_fish_ls.append(curr_perc[fish])
    
    ax.boxplot(all_fish_ls)
    ax.set_ylabel('richtige Entscheidungen in %')
    plt.xticks([1, 2, 3, 4, 5, 6], ['albi03', 'albi05', 'albi06', 'albi01', 'albi02', 'albi04'])
    ax.set_ylim([0, y_lims_single])
    ax.yaxis.set_ticks(np.arange(0, y_ticks_single, step=0.1))
    ax.set_title('Vergleich der Fische mit %s' % tag)
    
    #plt.savefig("/home/efish/PycharmProjects/philipp/figures/Boxplots_%s.png" %tag)
    #plt.savefig("/home/efish/PycharmProjects/philipp/figures/Boxplots_%s.svg" %tag)

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
    #ax.set_title('Vergleich der Reaktionszeiten im Testversuch, nur Albi05 und Albi06', fontsize=13)
    ax.set_title('Vergleich der Reaktionszeiten im Testversuch', fontsize=13)
    ax.boxplot(data)
    plt.xticks([1, 2, 3], ['hoch', 'niedrig', 'gemischt'], fontsize=12)
    ax.set_ylabel('Reaktionszeit in s', fontsize=12)

    #ttest analysis for each combination
    t_h_l = sc.stats.ttest_ind(high_react_ls, low_react_ls)
    t_h_m = sc.stats.ttest_ind(high_react_ls, mixed_react_ls)
    t_l_m = sc.stats.ttest_ind(low_react_ls, mixed_react_ls)
    print("T-test Ergebnisse für Reaktionszeiten: hoch + niedrig: %s, hoch + gemischt: %s, niedrig + gemischt: %s" %(t_h_l, t_h_m, t_l_m))


    #plt.savefig("/home/efish/PycharmProjects/philipp/figures/Vergleich_der_Reaktionszeiten_im_Testversuch_nur_05_06.png")
    #plt.savefig("/home/efish/PycharmProjects/philipp/figures/Vergleich_der_Reaktionszeiten_im_Testversuch.svg")
    #plt.savefig("/home/efish/PycharmProjects/philipp/figures/Vergleich_der_Reaktionszeiten_im_Testversuch.png")

    return plt


def diverse_statistics(data_mixed, data_high, data_low):

    # comparison between the different median of the fish for each stim
    high_test_perc = percentage_creation(data_high)
    low_test_perc = percentage_creation(data_low)
    mixed_test_perc = percentage_creation(data_mixed)

    for fish in high_test_perc:
        print("hochfrequenter Median von %s:" % fish, np.median(high_test_perc[fish]))

    for fish in low_test_perc:
        print("niederfrequenter Median von %s:" % fish, np.median(low_test_perc[fish]))

    for fish in mixed_test_perc:
        print("Gemischtfrequenter Median von %s:" % fish, np.median(mixed_test_perc[fish]))

    return print("stats done")


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
