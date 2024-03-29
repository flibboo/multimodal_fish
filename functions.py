from re import I
from time import time
from turtle import width
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import statistics
from scipy import stats
from scipy.stats import f_oneway
import scipy as sc
from IPython import embed
import matplotlib.gridspec as gridspec
import scikit_posthocs as sp
from scikit_posthocs import posthoc_tukey_hsd
import matplotlib.ticker as mtick




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

    # Farbcodierung für die verschiedenen Stimuli
    a01_colors = ['navy', 'cornflowerblue', 'blue']
    a02_colors = ['darkred', 'lightcoral', 'red']
    a03_colors = ['darkgreen', 'yellowgreen', 'springgreen']
    a04_colors = ['goldenrod', 'khaki', 'yellow']
    a05_colors = ['indigo', 'mediumpurple', 'blueviolet']
    a06_colors = ['darkmagenta', 'violet', 'fuchsia']

    # Farbversionen für Farbenblinde
    a01_colors = ['#00d290', '#00d290', '#00d290']
    a02_colors = ['#ff7ad4', '#ff7ad4', '#ff7ad4']
    a03_colors = ['#b22f10', '#b22f10', '#b22f10']
    a04_colors = ['#005cc7', '#005cc7', '#005cc7']
    a05_colors = ['#e5a771', '#e5a771', '#e5a771']
    a06_colors = ['#460052', '#460052', '#460052']

    linestyles_per_stim = ['dotted', 'dashed', 'dashdot']

    E_xy_sum = []
    E_x_sum = []
    E_y_sum = []
    E_x_2_sum = []
    m_summed = []
    b_summed = []

    color = "pink"
    if plot_name == "Alle Fische, hochfrequenter Stimulus (1000 Hz)":
        color = "black"
    if plot_name == "Alle Fische, niederfrequenter Stimulus (10 Hz)":
        color = "lightgrey"

    for index, fish in enumerate(all_fish):
        curr_data = percentages["perc_%s" % fish]
        time = len(curr_data)
        time_array = np.array(time)
        time_list = list(range(1, (time + 1)))

        ax.plot(time_list, curr_data, c="lightgrey" , linewidth=0.4)
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

        color_graph = locals()["a0%s_colors" % (index+1)]
        
        if plot_name == "Alle Fische, hochfrequenter Stimulus (1000 Hz)":
            color_graph = color_graph[0]
            linestylzz = linestyles_per_stim[0]
        elif plot_name == "Alle Fische, niederfrequenter Stimulus (10 Hz)":
            color_graph = color_graph[1]
            linestylzz = linestyles_per_stim[1]

        else:
            color_graph = "lightgrey"
        ax.plot(time_list, line_calc, c=color_graph, linewidth=1.5, linestyle=linestylzz)

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
    print("Lineare Regressionsgleichung gesammelter Plots:")
    print('y =', m_median, 'x +', b_median)
    # m_sum = (((N * E_xy_med) - (E_x_med * E_y_med)) / ((N * E_x_2_med) - (E_x_med * E_x_med)))
    # b_sum = (E_y_med - (m_sum * E_x_med)) / N

    line_calc = [m_median * x_l + b_median for x_l in time_list]  # loop is for multiplying lists

   
    # 50% line
    ax.axhline(0.5, linewidth= 0.5, linestyle='--', color="grey")
    ax.plot(time_list, line_calc, "black", linewidth=2)

    ax.set_xlabel('Tage', fontsize=12)
    ax.set_ylabel('Richtige Entscheidungen in %', fontsize=12)
    ax.set_xlim([0, (time + 1)])
    ax.set_ylim([0, 1.05])
    ax.yaxis.set_ticks(np.arange(0, 1.05, step=0.1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis='x', labelsize= 10)
    ax.tick_params(axis='y', labelsize= 10)
    #plt.title("%s" % plot_name)
    plt.savefig("/home/efish/PycharmProjects/philipp/figures/%s.svg" %plot_name)
    #plt.savefig("/home/efish/PycharmProjects/philipp/figures/%s.png" %plot_name)

    return plt


def plot_single(percentages, all_fish, plot_name_single, tag, binomial_dataframe_low, binomial_dataframe_high):
    # universal plot variables
    y_ticks_single = 1.05
    if tag == "use vertical lines":
        y_lims_single = 1.05
    else:    
        y_lims_single = 1.3
    
    # Farbcodierung für die verschiedenen Stimuli
    a01_colors = ['navy', 'cornflowerblue', 'blue']
    a02_colors = ['darkred', 'lightcoral', 'red']
    a03_colors = ['darkgreen', 'yellowgreen', 'springgreen']
    a04_colors = ['goldenrod', 'khaki', 'yellow']
    a05_colors = ['indigo', 'mediumpurple', 'blueviolet']
    a06_colors = ['darkmagenta', 'violet', 'fuchsia']

    # Farbversionen für Farbenblinde
    a01_colors = ['#00d290', '#00d290', '#00d290']
    a02_colors = ['#ff7ad4', '#ff7ad4', '#ff7ad4']
    a03_colors = ['#b22f10', '#b22f10', '#b22f10']
    a04_colors = ['#005cc7', '#005cc7', '#005cc7']
    a05_colors = ['#e5a771', '#e5a771', '#e5a771']
    a06_colors = ['#460052', '#460052', '#460052']

    linestyles_per_stim = ['dotted', 'dashed', 'dashdot']

    #Versuch alle 6 in einen Grid zu bekommen
    fig = plt.figure(figsize=(8, 10))
    outer = gridspec.GridSpec(3, 2, wspace=-0.1, hspace=0.1)
    if tag == "use vertical lines":
        outer = gridspec.GridSpec(3, 2, wspace=-0.1, hspace=0.2)

    for index, fish in enumerate(all_fish):
        # color coding
        

        color_graph = locals()["a0%s_colors" % (index+1)]
        
        if plot_name_single == ", hochfrequenter Stimulus (1000 Hz)":

            color_graph = color_graph[0]
            linestylzz = linestyles_per_stim[0]
        elif plot_name_single == ", niederfrequenter Stimulus (10 Hz)":

            color_graph = color_graph[1]
            linestylzz = linestyles_per_stim[1]

        else:
            color_graph = "lightgrey"

        i = index
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.1, hspace=0.1, width_ratios = (3, 1))
        j = 0

        curr_data = percentages["perc_%s" % fish]
        time = len(curr_data)
        time_list = list(range(1, (time + 1)))
        time_array = np.array(time_list)

        #fig, ax = plt.subplots(1 ,2, figsize=(10,8), gridspec_kw={'width_ratios': [4, 1]})
        ax = plt.Subplot(fig, inner[j])

        ax.plot(time_list, curr_data, c=color_graph, linestyle=linestylzz)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        code_pos_x = -4
        code_pos_y = 1.12
        if tag == "low use, no vert lines" or tag == "high use, no vert lines":
            code_pos_x = -2.85
            code_pos_y = 1.15
            if index == 0 or index == 1:
                ax.set_title("p-Werte über je 3 Tage", ha='center', va='center', fontsize=9)

            

        # for coding which fish is which on the graph
        fish_code = ["A","B","C","D","E","F"]
        ax.text(code_pos_x, code_pos_y, fish_code[index],  bbox=dict(facecolor='none', edgecolor='lightgrey', boxstyle='round,pad=0.2'), ha='center', va='center', fontweight = "bold", fontsize=13)

        y = curr_data
        x = time_list
        N = time

        # lineare Regression
        m_1, b_1 = np.polyfit(x, y, 1)
        
        m_1 = np.round(m_1, 3)
        b_1 = np.round(b_1, 3)
        print("Lineare Regression der einzel Plots:")
        print("%s: y = %s x + %s" % (fish, m_1, b_1))

        # 50% line + liniar Reg.line
        ax.axhline(0.5, linewidth= 0.5, linestyle='--', color="grey")
        ax.plot(x, m_1 * time_array + b_1, linewidth=1.5, c="black")

        # Pearson for R & p-Value
        r, p = np.round(sc.stats.pearsonr(x,y), 4)
        ax.text(15, 0.1, "R: %s  p: %s" % (r, p), ha='center', va='center', fontsize=6.5, c='darkorange')
        
        ax.set_xlim([0, 30])
        #ax.set_xlim([0, (time + 1)])
        ax.set_ylim([0, y_lims_single])
        ax.yaxis.set_ticks(np.arange(0, y_ticks_single, step=0.1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.tick_params(axis='x', labelsize= 10)
        ax.tick_params(axis='y', labelsize= 10)

        #ticks removal for the graphs who dont need them
        if (index % 2 != 0):
            ax.set_yticks([])
        else:
            ax.set_ylabel('Richtige Entscheidungen in %')
        if index < 4:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Tage', fontsize=12)
        #ax.set_title("%s %s" % (fish, plot_name_single))

        if tag == "use vertical lines":
            # if statements for different training days
            if fish == "2020albi01": fish_num = 15
            if fish == "2020albi02": fish_num = 19
            if fish == "2020albi03": fish_num = 20
            if fish == "2020albi04": fish_num = 16
            if fish == "2020albi05": fish_num = 19
            if fish == "2020albi06": fish_num = 16

            x = 7  # first training days
            ax.axvline(x, color="lightgrey", linestyle=':')  # first days mixed (7 days)
            ax.axvline(x=(fish_num + x), color="lightgrey", linestyle=':')  # only high training (equals fish_num)
            ax.axvline(x=(fish_num + x + 17), color="lightgrey", linestyle=':')  # only low training (17 days)
            ax.axvline(x=(fish_num + x + 17 + 3), color="lightgrey", linestyle=':')  # low + high training (3 days)
            
            ax.set_xlim([0, (time + 1)])


        # binomial visualisation, but only for high and low data
        f_p = 6.5
        f_stars = 7 
        text_pos_stars = 1.5
        text_pos_p = 2
        line_pos = 3.0
        if tag == "low use, no vert lines": # just used the tag because its only with high/low function
            binom_values = binomial_dataframe_low["bino_%s" %fish]
            for bv in binom_values:
                text = None
                if bv < 0.001:
                    text = '***'
                elif bv < 0.01:
                    text = '**'
                elif bv < 0.05:
                    text = "*"
                else:
                    text = "n.s."
                ax.axvline(line_pos, ymin = 0.8, ymax = 0.82)
                ax.text(text_pos_stars, 1.05, text, ha='center', va='center', fontsize=f_stars)
                ax.text(text_pos_p, 1.18 , np.round(bv,3), rotation= 45, ha='center', va='center', fontsize=f_p)
                text_pos_p += 3  # update text x position for next binom value
                text_pos_stars += 3
                line_pos += 3
                                      
        
           
        if tag == "high use, no vert lines": 
 
            binom_values = binomial_dataframe_high["bino_%s" %fish]
            for bv in binom_values:
                text = None
                if bv < 0.001:
                    text = '***'
                elif bv < 0.01:
                    text = '**'
                elif bv < 0.05:
                    text = "*"
                else:
                    text = "n.s."
                ax.axvline(line_pos, ymin = 0.8, ymax = 0.82)
                ax.text(text_pos_stars, 1.05, text, ha='center', va='center', fontsize=f_stars)
                ax.text(text_pos_p, 1.18 , np.round(bv,3), rotation= 45, ha='center', va='center', fontsize=f_p)
                text_pos_p += 3  # update text x position for next binom value
                text_pos_stars += 3
                line_pos += 3
                                      
        fig.add_subplot(ax)
        
        # boxplot 
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        j = 1
        ax = plt.Subplot(fig, inner[j])

        bp = ax.boxplot(curr_data, vert=True, patch_artist=False)
        ax.get_xaxis().set_ticks([])
        #ax[1].yaxis.set_label_position("right")
        #ax[1].yaxis.tick_right()
        #ax[1].set_ylabel('correct choices in %')
        #ax[1].yaxis.set_ticks(np.arange(0, y_ticks_single, step=0.1))
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0, y_lims_single])
        ax.set_facecolor('none')
        ax.axis('off')
        
        for medline in bp['medians']:
            linedata = medline.get_ydata()
            median = linedata[0]
            print("Boxplotmedian: %s" % median)

        for patch in bp['medians']:
            patch.set_color('black')

        """
        q1 = [min(item.get_ydata()) for item in bp['boxes']]
        q3 = [max(item.get_ydata()) for item in bp['boxes']]
        print(f'Q1: {q1}\n' f'Q3: {q3}')
        iqa = q3[0] - q1[0]
        print("Boxplot IQA = %s" % iqa)

        """
        fig.add_subplot(ax)
    
    
    ax.set_xlabel('Tage', fontsize = 12)
    ax.set_ylabel('Richtige Entscheidungen in %', fontsize = 12)
    ax.yaxis.set_ticks(np.arange(0, 1.05, step=0.1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    if plot_name_single == ", niederfrequenter Stimulus (10 Hz)":
        plt.savefig("/home/efish/PycharmProjects/philipp/figures/low_stim_sixer.svg")
    elif plot_name_single == ", hochfrequenter Stimulus (1000 Hz)":
        plt.savefig("/home/efish/PycharmProjects/philipp/figures/high_stim_sixer.svg")
    else:
        plt.savefig("/home/efish/PycharmProjects/philipp/figures/all_stim_sixer.svg")

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
    only_06 = "n"
    for fish in high_test_perc:
        if fish == "perc_2020albi06":# or fish == "perc_2020albi06": # can be skipped, if all fish should be included
            only_06 = "y" # only for the use of changing color and saving when only albi06 is used
            yr_highness.extend(high_test_perc[fish])

    yr_lowness = []
    for fish in low_test_perc:
        if fish == "perc_2020albi06":# or fish == "perc_2020albi06":
            yr_lowness.extend(low_test_perc[fish])

    yr_mixedness = []
    for fish in mixed_test_perc:
        if fish == "perc_2020albi06":# or fish == "perc_2020albi06":
            yr_mixedness.extend(mixed_test_perc[fish])

    #yr_mixedness = np.array(yr_mixedness)
    #yr_mixedness = yr_mixedness[yr_mixedness > 0.3]

    data = [yr_highness, yr_lowness, yr_mixedness]
    fig, ax = plt.subplots(figsize=(10,8))

    #ax.set_title('Vergleich der Stimuli im Testversuch')

    # 50% line
    ax.axhline(0.5, linewidth= 0.5, linestyle='--', color="grey")

    print("Boxplot des Testing:")

    bp = ax.boxplot(data, vert=True, patch_artist=False)
    #print("medians of boxplots: %s" % bp['medians'][0].get_ydata())
    for medline in bp['medians']:
        linedata = medline.get_ydata()
        median = linedata[0]
        print("Boxplotmedian: %s" % median)
    
    q1 = [min(item.get_ydata()) for item in bp['boxes']]
    q3 = [max(item.get_ydata()) for item in bp['boxes']]
    print(f'Q1: {q1}\n' f'Q3: {q3}')
    for q1v, q3v  in zip(q1, q3):
        iqa = q3v - q1v
        print("Boxplot IQA = %s" % iqa)

    anova_results = f_oneway(yr_highness, yr_lowness, yr_mixedness)

    print("Anova: {}".format(anova_results))

    hoc_names = [["high"]*len(yr_highness), ["low"]*len(yr_lowness), ["kombiniert"]*len(yr_mixedness)]
    hoc_list =[yr_highness, yr_lowness, yr_mixedness]
    posti = sp.posthoc_tukey_hsd(np.concatenate(hoc_list), np.concatenate(hoc_names))
    print("Posthoc for Testing: %s" %posti)

    bp = ax.boxplot(data, vert=True, patch_artist=True)
    if only_06 == "y":
        #colors = ['darkmagenta', 'violet', 'fuchsia']
        colors = ['#460052', '#460052', '#460052'] # farbenblinden Version
        linestyles_per_stim = ['dotted', 'dashed', 'dashdot']
    else:    
        colors = ['black','lightgrey', 'grey']
        linestyles_per_stim =['dotted', 'dashed', 'dashdot']

    holder = 0

    for patch, color in zip(bp['boxes'], colors):
        patch.set(color = color, facecolor='white', linestyle= linestyles_per_stim[holder], linewidth=2.5)
        holder += 1
    
    for patch, color in zip(bp['medians'], colors):
        patch.set_color('black')
    
    plt.xticks([1, 2, 3], ['hoch', 'niedrig', 'gemischt'], fontsize=10)
    ax.set_ylabel('Richtige Entscheidungen in %', fontsize=12)
    ax.yaxis.set_ticks(np.arange(0, 1.05, step=0.1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel('Stimulus', fontsize=12)
    
    if only_06 == 'y':
        #plt.savefig("/home/efish/PycharmProjects/philipp/figures/Vergleich_der_Stimuli_im_Testversuch_nur_06.png")
        plt.savefig("/home/efish/PycharmProjects/philipp/figures/Vergleich_der_Stimuli_im_Testversuch_nur_06.svg")
    else:
        plt.savefig("/home/efish/PycharmProjects/philipp/figures/Vergleich_der_Stimuli_im_Testversuch.svg")
        #plt.savefig("/home/efish/PycharmProjects/philipp/figures/Vergleich_der_Stimuli_im_Testversuch.png")


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
    fix, ax = plt.subplots(figsize=(10, 8))
    all_fish_ls = []
    for fish in curr_perc:
        all_fish_ls.append(curr_perc[fish])
    
    #ax.boxplot(all_fish_ls)
    ax.set_ylabel('Richtige Entscheidungen in %', fontsize=12)
    ax.set_ylim([0, y_lims_single])
    ax.yaxis.set_ticks(np.arange(0, y_ticks_single, step=0.1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    #ax.set_title('Vergleich der Fische mit %s' % tag)
    
    print("Einzelne Plotwerte von %s" % tag)

    bp = ax.boxplot(all_fish_ls, vert=True, patch_artist=False)
    for medline in bp['medians']:
        linedata = medline.get_ydata()
        median = linedata[0]
        print("Boxplotmedian: %s" % median)

    q1 = [min(item.get_ydata()) for item in bp['boxes']]
    q3 = [max(item.get_ydata()) for item in bp['boxes']]
    print(f'Q1: {q1}\n' f'Q3: {q3}')
    for q1v, q3v  in zip(q1, q3):
        iqa = q3v - q1v
        print("Boxplot IQA = %s" % iqa)

    # statistische Analyse ob sie wirklich unterschiedlich sind
    anova_results =  f_oneway(all_fish_ls[0], all_fish_ls[1], all_fish_ls[2], all_fish_ls[3], all_fish_ls[4], all_fish_ls[5]) # sagt ob was unterschiedlich ist mit p-wert
    print("Anova - %s: %s" % (tag, anova_results))

    hoc_names = [['albi01'] * len(all_fish_ls[0]),
                 ['albi02'] * len(all_fish_ls[1]),
                 ['albi03'] * len(all_fish_ls[2]),
                 ['albi04'] * len(all_fish_ls[3]),
                 ['albi05'] * len(all_fish_ls[4]),
                 ['albi06'] * len(all_fish_ls[5])]
    hoc_list = [all_fish_ls[0], all_fish_ls[1], all_fish_ls[2], all_fish_ls[3], all_fish_ls[4], all_fish_ls[5]]

    posti = sp.posthoc_tukey_hsd(np.concatenate(hoc_list), np.concatenate(hoc_names))


    hoc_names = [['albi01'], ['albi02'], ['albi03'], ['albi04'], ['albi05'], ['albi06']]
    #posti = sp.posthoc_tukey_hsd(np.concatenate(hoc_array), np.concatenate(hoc_names)) # gibt tabelle 1 = signif. untersch/ 0 = ne
    print("Posthoc test - %s: %s" %(tag, posti))

    a01_colors = ['navy', 'cornflowerblue', 'blue']
    a02_colors = ['darkred', 'lightcoral', 'red']
    a03_colors = ['darkgreen', 'yellowgreen', 'springgreen']
    a04_colors = ['goldenrod', 'khaki', 'yellow']
    a05_colors = ['indigo', 'mediumpurple', 'blueviolet']
    a06_colors = ['darkmagenta', 'violet', 'fuchsia']

    # Farbversionen für Farbenblinde
    a01_colors = ['#00d290', '#00d290', '#00d290']
    a02_colors = ['#ff7ad4', '#ff7ad4', '#ff7ad4']
    a03_colors = ['#b22f10', '#b22f10', '#b22f10']
    a04_colors = ['#005cc7', '#005cc7', '#005cc7']
    a05_colors = ['#e5a771', '#e5a771', '#e5a771']
    a06_colors = ['#460052', '#460052', '#460052']

    linestyles_per_stim = ['dotted', 'dashed', 'dashdot']


    colors = []
    bp = ax.boxplot(all_fish_ls, vert=True, patch_artist=True)
    if tag == "hochfrequentem Stimulus (Testdaten)" or tag == "hochfrequentem Stimulus (Trainingsdaten)":
        for index, fish in enumerate(all_fish_ls):
            color_graph = locals()["a0%s_colors" % (index+1)]
            colors.append(color_graph[0])
            linestylzz = linestyles_per_stim[0]
    if tag == "niederfrequentem Stimulus (Testdaten)"or tag == "niederfrequentem Stimulus (Trainingsdaten)":
        for index, fish in enumerate(all_fish_ls):
            color_graph = locals()["a0%s_colors" % (index+1)]
            colors.append(color_graph[1])
            linestylzz = linestyles_per_stim[1]
    if tag == "gemischtem Stimulus (Testdaten)":
        for index, fish in enumerate(all_fish_ls):
            color_graph = locals()["a0%s_colors" % (index+1)]
            colors.append(color_graph[2])
            linestylzz = linestyles_per_stim[2]


    for patch, colorus in zip(bp['boxes'], colors):
        patch.set(color = colorus, facecolor = 'white', linestyle=linestylzz, linewidth= 2.5)
    
    for patch, colorus in zip(bp['medians'], colors):
        patch.set(color='black')

    #ändern der outlines der Boxen
    #boxprops = dict(linestyle='--', linewidth=5, color='darkgoldenrod')

    # 50% line
    ax.axhline(0.5, linewidth= 0.5, linestyle='--', color="grey")

    
    plt.xticks([1, 2, 3, 4, 5, 6], ['albi01', 'albi02', 'albi03', 'albi04', 'albi05', 'albi06'], fontsize=10)
    ax.set_xlabel('Fische', fontsize=12)
    


    #plt.savefig("/home/efish/PycharmProjects/philipp/figures/Boxplots_%s.png" %tag)
    plt.savefig("/home/efish/PycharmProjects/philipp/figures/Boxplots_%s.svg" %tag)

    return plt


def reaction_time_analysis(times, data, stim):
    # each frame contains a needed information: the data, the reaction times and the used stimuli
    high_react_ls = []
    low_react_ls = []
    mixed_react_ls = []
    only_06 = 'n'
    for fish in data.columns:
        #if fish == "2020albi06":# or fish == "2020albi06": # can be skipped, if all fish should be included
            #only_06 = 'y'
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
    fig, ax = plt.subplots(figsize=(10, 8))
    #ax.set_title('Vergleich der Reaktionszeiten im Testversuch, nur Albi05 und Albi06', fontsize=13)
    #ax.set_title('Vergleich der Reaktionszeiten im Testversuch', fontsize=13)
    ax.set_ylabel('Reaktionszeit in s', fontsize=12)
    ax.set_xlabel('Stimulus', fontsize=12)

    print("Reaktionszeiten Werte & Angaben:")
    bp = ax.boxplot(data, vert=True, patch_artist=False)
    for medline in bp['medians']:
        linedata = medline.get_ydata()
        median = linedata[0]
        print("Boxplotmedian: %s" % median)
    
    q1 = [min(item.get_ydata()) for item in bp['boxes']]
    q3 = [max(item.get_ydata()) for item in bp['boxes']]
    print(f'Q1: {q1}\n' f'Q3: {q3}')
    for q1v, q3v  in zip(q1, q3):
        iqa = q3v - q1v
        print("Boxplot IQA = %s" % iqa)

    anova_results = f_oneway(high_react_ls, low_react_ls, mixed_react_ls)

    print("Anova: {}".format(anova_results))
    hoc_names = [["high"]*len(high_react_ls), ["low"]*len(low_react_ls), ["kombiniert"]*len(mixed_react_ls)]
    hoc_list =[high_react_ls, low_react_ls, mixed_react_ls]
    posti = sp.posthoc_tukey_hsd(np.concatenate(hoc_list), np.concatenate(hoc_names))
    print("Posthoc for Testing React: %s" %posti)



    bp = ax.boxplot(data, vert=True, patch_artist=True)

    if only_06 == "y":
        #colors = ['darkmagenta', 'violet', 'fuchsia']
        colors = ['#460052', '#460052', '#460052']
        linestyles_per_stim =['dotted', 'dashed', 'dashdot']
    else:    
        colors = ['black','lightgrey', 'grey']
        linestyles_per_stim =['dotted', 'dashed', 'dashdot']

    holder = 0
    for patch, color in zip(bp['boxes'], colors):
        patch.set(color = color, facecolor = 'white',linestyle=linestyles_per_stim[holder], linewidth= 2.5)
        holder += 1
    
    for patch, color in zip(bp['medians'], colors):
        patch.set_color('black')
    
    plt.xticks([1, 2, 3], ['hoch', 'niedrig', 'gemischt'], fontsize=10)
    #ttest analysis for each combination
    t_h_l = sc.stats.ttest_ind(high_react_ls, low_react_ls)
    t_h_m = sc.stats.ttest_ind(high_react_ls, mixed_react_ls)
    t_l_m = sc.stats.ttest_ind(low_react_ls, mixed_react_ls)
    print("T-test Ergebnisse für Reaktionszeiten: hoch + niedrig: %s, hoch + gemischt: %s, niedrig + gemischt: %s" %(t_h_l, t_h_m, t_l_m))

    if only_06 == 'y':
        plt.savefig("/home/efish/PycharmProjects/philipp/figures/Vergleich_der_Reaktionszeiten_im_Testversuch_nur_06.svg")
    else:
        plt.savefig("/home/efish/PycharmProjects/philipp/figures/Vergleich_der_Reaktionszeiten_im_Testversuch.svg")
    #plt.savefig("/home/efish/PycharmProjects/philipp/figures/Vergleich_der_Reaktionszeiten_im_Testversuch.png")

    return plt


def diverse_statistics(data_mixed, data_high, data_low):

    # comparison between the different median of the fish for each stim
    high_test_perc = percentage_creation(data_high)
    low_test_perc = percentage_creation(data_low)
    mixed_test_perc = percentage_creation(data_mixed)
    
    print("Test Means:") # vllt doch median, da sonst immer genutzt
    for fish in high_test_perc:
        print("hochfrequenter Mean von %s:" % fish, np.mean(high_test_perc[fish]))

    for fish in low_test_perc:
        print("niederfrequenter Mean von %s:" % fish, np.mean(low_test_perc[fish]))

    for fish in mixed_test_perc:
        print("Gemischtfrequenter Mean von %s:" % fish, np.mean(mixed_test_perc[fish]))

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
        #plt.title("%s %s" % (fish, plot_name_single))
        plt.plot(x_axis, model.predict_proba(x_axis)[:, 1])

    return plt
