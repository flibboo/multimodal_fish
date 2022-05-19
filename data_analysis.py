from matplotlib import testing
import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed
import numpy as np
from functions import *

"""
# Open file
"""
# training files + sorting
training_data = pd.read_hdf("training_dataframe.hf")
training_data = training_data.sort_index(axis=1)
training_low_data = pd.read_hdf("training_low_dataframe.hf")
training_low_data = training_low_data.sort_index(axis=1)
training_high_data = pd.read_hdf("training_high_dataframe.hf")
training_high_data = training_high_data.sort_index(axis=1)

# testing files + sorting
testing_data = pd.read_hdf("testing_dataframe.hf")
testing_data = testing_data.sort_index(axis=1)
testing_low_data = pd.read_hdf("testing_low_dataframe.hf")
testing_low_data = testing_low_data.sort_index(axis=1)
testing_high_data = pd.read_hdf("testing_high_dataframe.hf")
testing_high_data = testing_high_data.sort_index(axis=1)
testing_mixed_data = pd.read_hdf("testing_mixed_dataframe.hf")
testing_mixed_data = testing_mixed_data.sort_index(axis=1)
testing_react_times = pd.read_hdf("testing_reaction_time_dataframe.hf")
testing_react_times = testing_react_times.sort_index(axis=1)
corr_test_react_times = pd.read_hdf("correct_testing_react_time_dataframe.hf")
corr_test_react_times = corr_test_react_times.sort_index(axis=1)
testing_stim_frame = pd.read_hdf("testing_stim_frame.hf")
testing_stim_frame = testing_stim_frame.sort_index(axis=1)

all_fish = np.array(training_data.columns)
percentages = percentage_creation(training_data)
binomial_dataframe_low, binomial_dataframe_high = binomial_data(training_low_data, training_high_data)


"""
# all fish - all data -> dient nur noch zur Anschauung, keinen Wert bei Auswertung
"""
# fish plots
"""
plot_name = "Alle Fische, hoch- und niederfrequente Stimuli"
plot_all_together(percentages, all_fish, plot_name)
plt.show()
#plt.close()
"""
"""
plot_name_single = ", hoch- und niederfrequente Stimuli"
tag = "use vertical lines" # this tag is for filtering out a graphic add, which we only need here
plot_single(percentages, all_fish, plot_name_single, tag, binomial_dataframe_low, binomial_dataframe_high)
plt.show()


"""
# using only low/high data
"""

plot_name = "Alle Fische, niederfrequenter Stimulus (10 Hz)"
plot_name_single = ", niederfrequenter Stimulus (10 Hz)"
low_data_use(training_low_data, all_fish, plot_name, plot_name_single, binomial_dataframe_low, binomial_dataframe_high)

tag = "niederfrequentem Stimulus (Trainingsdaten)"
boxplotting_for_singles(training_low_data, tag)
plt.show()

plot_name = "Alle Fische, hochfrequenter Stimulus (1000 Hz)"
plot_name_single = ", hochfrequenter Stimulus (1000 Hz)"
high_data_use(training_high_data, all_fish, plot_name, plot_name_single, binomial_dataframe_low, binomial_dataframe_high)

tag = "hochfrequentem Stimulus (Trainingsdaten)"
boxplotting_for_singles(training_high_data, tag)
plt.show()
"""
"""
# testing analysis
"""
boxplotting(testing_high_data, testing_low_data, testing_mixed_data)
plt.show()

reaction_time_analysis(testing_react_times, testing_data, testing_stim_frame)
plt.show()

tag = "hochfrequentem Stimulus (Testdaten)"
boxplotting_for_singles(testing_high_data, tag)
plt.show()

tag = "niederfrequentem Stimulus (Testdaten)"
boxplotting_for_singles(testing_low_data, tag)
plt.show()

tag = "gemischtem Stimulus (Testdaten)"
boxplotting_for_singles(testing_mixed_data, tag)
plt.show()

"""
# other statistics
"""
diverse_statistics(testing_mixed_data, testing_high_data, testing_low_data)

"""
"""
# logistic regression -> ich bin leider eine Enttäuschung
"""
# low data use
plot_name_single = ", logistische Regression mit niederfrequentem Stimulus"
for fish in all_fish:
    flattened_fish = flatten_fish(fish, training_low_data)
    fish_regression(fish, flattened_fish, percentages, plot_name_single)
    plt.show()
    #plt.close()

# high data use
plot_name_single = ", logistic Regression mit hochfrequentem Stimulus"
for fish in all_fish:
    flattened_fish = flatten_fish(fish, training_high_data)
    fish_regression(fish, flattened_fish, percentages, plot_name_single)
    plt.show()
    #plt.close()
"""