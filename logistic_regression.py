import pandas as pd
from IPython import embed
import numpy as np
from functions import *
# Open file
training_data = pd.read_hdf("training_dataframe.hf")
training_low_data = pd.read_hdf("training_low_dataframe.hf")
training_high_data = pd.read_hdf("training_high_dataframe.hf")

all_fish = np.array(training_data.columns)

percentages = percentage_creation(training_data)

# fish plots
plot_all_together(percentages, all_fish)
plt.show()

plot_single(percentages, all_fish)
plt.show()


# Parse through fish
for fish in all_fish:
    flattened_fish = flatten_fish(fish, training_data) # hier werden alle daten verwendet, nicht nur high oder low

    # At this point you can do the Logistic Regression

    fish_regression(fish, flattened_fish, percentages)


# using only data of low/high stimuli (noch in Probephase)

low_data_use(training_low_data, all_fish) # funktioniert nicht wirklich, die gesammelten funktionen sind mist

high_data_use(training_high_data, all_fish)