import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed
import numpy as np
from functions import *

"""
Open file
"""
# training files
training_data = pd.read_hdf("training_dataframe.hf")
training_low_data = pd.read_hdf("training_low_dataframe.hf")
training_high_data = pd.read_hdf("training_high_dataframe.hf")

# testing files
testing_data = pd.read_hdf("testing_dataframe.hf")
testing_low_data = pd.read_hdf("testing_low_dataframe.hf")
testing_high_data = pd.read_hdf("testing_dataframe.hf")
testing_mixed_data = pd.read_hdf("testing_dataframe.hf")

all_fish = np.array(training_data.columns)
percentages = percentage_creation(training_data)

"""
all fish - all data
"""
# fish plots
plot_name = "All fish, high and low frequent stimuli"
plot_all_together(percentages, all_fish, plot_name)
#plt.show()
plt.close()

plot_name_single = ", high and low frequent stimuli"
plot_single(percentages, all_fish, plot_name_single)
plt.show()
#plt.close()

"""
using only low/high data
"""

plot_name = "All fish, low frequent stimuli"
plot_name_single = ", low frequent stimuli"
low_data_use(training_low_data, all_fish, plot_name, plot_name_single)

plot_name = "All fish, high frequent stimuli"
plot_name_single = ", high frequent stimuli"
high_data_use(training_high_data, all_fish, plot_name, plot_name_single)

"""
logistic regression
"""
# low data use
plot_name_single = ", logistic regression with low frequent stimuli"
for fish in all_fish:
    flattened_fish = flatten_fish(fish, training_low_data)
    fish_regression(fish, flattened_fish, percentages, plot_name_single)
    #plt.show()
    plt.close()

# high data use
plot_name_single = ", logistic regression with high frequent stimuli"
for fish in all_fish:
    flattened_fish = flatten_fish(fish, training_high_data)
    fish_regression(fish, flattened_fish, percentages, plot_name_single)
    #plt.show()
    plt.close()

"""
testing analysis
"""
#boxplotting(testing_high_data, testing_low_data, testing_mixed_data)

#plt.show()




"""
other statistics
"""
diverse_statistics(percentages, flattened_fish)
