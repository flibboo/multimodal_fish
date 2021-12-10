import matplotlib.pyplot as plt
import numpy as np
from itertools import repeat
from scipy import stats
from IPython import embed
import statistics
from scipy.optimize import curve_fit
import os as os 
from sklearn.linear_model import LogisticRegression

curr_filepath = os.getcwd()
fig_filepath_base = os.path.dirname(curr_filepath)
fig_filepath= os.path.join(fig_filepath_base, 'figures') 
if not os.path.exists(fig_filepath):
    os.mkdir(fig_filepath)

# variables
threshold = list(repeat(0.80, 18))
midline = list(repeat(0.50, 18))
time = list(range(1, 17))
time_array = np.array(time)

# variables for all data
threshold = list(repeat(0.80, 24))
midline = list(repeat(0.50, 24))
time = list(range(1, 24))
time_array = np.array(time)

# data without pre-phase
"""
a01_corr = np.array([7, 10, 13, 11, 9, 11, 6, 11, 11, 11, 12, 13, 14, 13, 12, 14])
a01_trials = np.array([15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15])

a02_corr = np.array([3, 7, 6, 5, 6, 11, 8, 6, 10, 7, 13, 10, 13, 13, 8, 9])
a02_trials = np.array([15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 15, 15, 15, 15])

a03_corr = np.array([9, 11, 7, 9, 10, 8, 9, 10, 7, 9, 9, 12, 9, 11, 9, 8])
a03_trials = np.array([15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15])

a04_corr = np.array([10, 10, 13, 9, 10, 10, 10, 13, 13, 13, 14, 12, 14, 10, 15, 11])
a04_trials = np.array([15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15])

a05_corr = np.array([9, 7, 9, 9, 7, 8, 2, 6, 5, 3, 5, 6, 12, 8, 10, 13])
a05_trials = np.array([15, 11, 15, 15, 14, 13, 6, 12, 10, 5, 14, 14, 15, 12, 15, 15])

a06_corr = np.array([9, 8, 11, 8, 11, 9, 10, 9, 14, 9, 11, 13, 13, 12, 12, 12])
a06_trials = np.array([15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15])
"""
# data with pre-phase

a01_corr = np.array([17, 13, 9, 7, 10, 10, 9, 7, 10, 13, 11, 9, 11, 6, 11, 11, 11, 12, 13, 14, 13, 12, 14])
a01_trials = np.array([26, 26, 15, 12, 16, 13, 17, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15])

a02_corr = np.array([8, 10, 6, 12, 6, 5, 12, 3, 7, 6, 5, 6, 11, 8, 6, 10, 7, 13, 10, 13, 13, 8, 9])
a02_trials = np.array([24, 22, 14, 21, 16, 12, 17, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 15, 15, 15, 15])

a03_corr = np.array([12, 15, 11, 14, 7, 6, 10, 9, 11, 7, 9, 10, 8, 9, 10, 7, 9, 9, 12, 9, 11, 9, 8])
a03_trials = np.array([24, 26, 15, 25, 17, 13, 17, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15])

a04_corr = np.array([7, 9, 8, 20, 8, 13, 4, 10, 10, 13, 9, 10, 10, 10, 13, 13, 13, 14, 12, 14, 10, 15, 11])
a04_trials = np.array([15, 14, 13, 25, 12, 16, 7, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15])

a05_corr = np.array([2, 6, 5, 7, 5, 7, 4, 9, 7, 9, 9, 7, 8, 2, 6, 5, 3, 5, 6, 12, 8, 10, 13])
a05_trials = np.array([9, 13, 13, 13, 10, 13, 7, 15, 11, 15, 15, 14, 13, 6, 12, 10, 5, 14, 14, 15, 12, 15, 15])

a06_corr = np.array([6, 6, 7, 16, 10, 4, 10, 9, 8, 11, 8, 11, 9, 10, 9, 14, 9, 11, 13, 13, 12, 12, 12])
a06_trials = np.array([14, 14, 13, 25, 17, 7, 17, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15])

names = ['a01', 'a02','a03','a04','a05','a06']

# convert to percent
correct_choices = [a01_corr, a02_corr, a03_corr, a04_corr, a05_corr, a06_corr]
all_trials = [a01_trials, a02_trials, a03_trials, a04_trials, a05_trials, a06_trials]

all_percentages = []

for choice, trial in zip(correct_choices, all_trials):
    all_percentages.append((choice/trial))

print(all_percentages)

# statistics
day_1 = []
day_16 = []
for percentage in all_percentages:
    print(stats.shapiro(percentage))  # Shapiro-Wilk-Test
    print(np.corrcoef(time, percentage))  # Pearson Correlation
    print(stats.spearmanr(time, percentage, axis=1))  # Spearman Correlation   TIME CORRECT?! want int instead of list
    day_1.append(percentage[0])
    day_16.append(percentage[-1])
print(stats.ttest_rel(day_1, day_16))

# plotting
# all plots in one graphic
fig, ax = plt.subplots()
#ax.set_title('albi06')

E_xy_sum = []
E_x_sum = []
E_y_sum = []
E_x_2_sum = []

for percentage in all_percentages:
    ax.plot(time, percentage, "lightgrey", linewidth=0.8)
    x = time_array
    N = 16
    N = 23 # for all data

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

    # t-test?
    #print('albi01', stats.ttest_rel(m, b))

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

ax.plot(range(0, 24), threshold, '--', linewidth=0.8)  # 80% Grenze
ax.plot(range(0, 24), midline, '--', linewidth=0.8)  # 50% Grenze

ax.set_xlabel('days')

ax.set_ylabel('correct choices in %')
ax.set_xlim([0, 24])
ax.set_ylim([0, 1.05])

# adjusting the steps on the axes

plt.yticks([0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
#plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]) # for all data


#plt.show()
plt.close()

# each fish got its own plot
for percentage, name in zip(all_percentages, names):
    x = time_array
    y = percentage
    fig, ax = plt.subplots()
    ax.plot(time, percentage)

    #ax.plot(range(0, 18), threshold, '--', linewidth=0.8)  # 80% Grenze
    #ax.plot(range(0, 18), midline, '--', linewidth=0.8)  # 50% Grenze
    # for all data
    ax.plot(range(0, 24), threshold, '--', linewidth=0.8)  # 80% Grenze
    ax.plot(range(0, 24), midline, '--', linewidth=0.8)  # 50% Grenze

    ax.set_xlabel('days')

    ax.set_ylabel('correct choices in %')
    ax.set_xlim([0, 17])
    ax.set_ylim([0, 0.105])

    # adjusting the steps on the axes

    plt.yticks([0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    #plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]) # for all data

    # regression line (professional)

    #m_1, b_1 = np.polyfit(x, y, 1)
    #ax.plot(x, m_1*time_array + b_1)

    # linear regression (handmade)

    x = time_array
    N = 16
    N = 23 # for all data

    y = percentage
    E_xy = sum(time_array * percentage)
    E_x = sum(time_array)
    E_y = sum(percentage)
    E_x_2 = sum(x * x)
    m = (((N * E_xy) - (E_x * E_y)) / ((N * E_x_2) - (E_x * E_x)))
    b = (E_y - (m * E_x)) / N
    ax.plot(time, m*time_array+b)
    print('y =', m, 'x +', b)

    plt.savefig(os.path.join(fig_filepath, '%s.png' %name), dpi=400)
    plt.close()


for percentage, name in zip(all_percentages, names):
    x_axis = np.arange(len(time)).reshape(-1, 1)
    bool_trial = percentage>0.7
    y_axis = bool_trial*1

model = LogisticRegression(solver='liblinear', random_state=0)
# finish logistic regression