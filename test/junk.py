""" Time series from Cousera"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# print(f'tensorflow version: {tf.__version__}')


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


'Trend and Seasonality'


def trend(tme, slope=0.0):
    return slope * tme


'create a time series that just trends upward:'

time = np.arange(4 * 365 + 1)
baseline = 10
series = trend(time, 0.1)
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()
