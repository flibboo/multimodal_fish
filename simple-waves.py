# i will create great visuals
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
import numpy as np                   # for multi-dimensional containers
import pandas as pd                  # for DataFrames
import plotly.graph_objects as go    # for data visualisation
import plotly.io as pio              # to set shahin plot layout



"""
time        = np.arange(0, 10, (1/1000));
amplitude   = np.sin(time)
plt.plot(time, amplitude)
plt.title('Sine wave')
plt.xlabel('Time')
plt.ylabel('Amplitude = sin(time)')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.show()
"""

sample_rate = 100000
start_time = 0
end_time = 0.2

time = np.arange(start_time, end_time, 1/sample_rate)

frequency = 1000
amplitude = 1
theta = 0
sinewave_h = amplitude * np.sin(2 * np.pi * frequency * time + theta)

frequency = 10
sinewave_l = amplitude * np.sin(2 * np.pi * frequency * time + theta)

fig, (ax1, ax2) = plt.subplots(2 ,figsize=(10,8))
#ax1.set_xlabel("Zeit in s")
ax1.set_ylabel("Amplitude in mV/cm", fontsize = 12)
ax1.plot(time, sinewave_h, linewidth = 2)
ax1.set_xticks([])
ax1.yaxis.set_ticks(np.arange(-1, 1.01, step=0.25))
ax1.tick_params(axis='x', labelsize= 10)
ax1.tick_params(axis='y', labelsize= 10)

ax2.set_xlabel("Zeit in s", fontsize = 12)
ax2.set_ylabel("Amplitude in mV/cm", fontsize= 12)
ax2.plot(time, sinewave_l, linewidth = 2)
ax2.yaxis.set_ticks(np.arange(-1, 1.01, step=0.25))
ax2.xaxis.set_ticks(np.arange(0, 0.21, step=0.02))
ax2.tick_params(axis='x', labelsize= 10)
ax2.tick_params(axis='y', labelsize= 10)
plt.savefig("/home/efish/PycharmProjects/philipp/figures/sinewaves.svg")

plt.show()