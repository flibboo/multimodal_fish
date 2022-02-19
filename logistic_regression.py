import pandas as pd
from IPython import embed
import numpy as np
from functions import *
# Open file
training_data = pd.read_hdf("training_dataframe.hf")
all_fish = np.array(training_data.columns)


percentages = percentage_creation(training_data)

# Parse through fish
for fish in all_fish:
    flattened_fish = flatten_fish(fish, training_data)

    # At this point you can do the Logistic Regression

    fish_regression(fish, flattened_fish, percentages)
