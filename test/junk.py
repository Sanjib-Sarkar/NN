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
# plt.show()

'generate a time series with a seasonal pattern:'


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

baseline = 10
amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
# plt.show()

"Now let's create a time series with both trend and seasonality:"
slope = 0.05
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
# plt.show()


# "Noise"

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

noise_level = 5
noise = white_noise(time, noise_level, seed=42)

plt.figure(figsize=(10, 6))
plot_series(time, noise)
# plt.show()
" Add white noise"
series += noise

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

"""ll right, this looks realistic enough for now. Let's try to forecast it.
 We will split it into two periods: the training period and the validation period 
 (in many cases, you would also want to have a test period). The split will be at time step 1000."""