import time
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt

filepath = r"C:\Users\w10122210\The University of Southern Mississippi\Ocean Exploration Lab - Documents\Data\20220226 WAM-V lawn mower training sets"
# files = [f for f in listdir(filepath) if isfile(join(filepath, f))]
files = [f for f in listdir(filepath) if f.endswith('.csv')]
labels = files
# fig, axes = plt.subplots(len(files), 1, figsize=(9, 11), tight_layout=True)
# for file, ax, label in zip(files, axes, labels):
#     file = join(filepath, file)
#     df = pd.read_csv(file, skiprows=1)
#     df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng'}, errors="raise")
#     df = df.loc[(df['lat'] != 0)]
#     ax.plot(df['lng'], df['lat'], label=label)
#     ax.legend()
#     time.sleep(1)
#
# plt.show()


def plot_series(x, y, format="-", start=0, end=None, xlabel='x', ylabel='y', label='plot'):
    plt.plot(x[start:end], y[start:end], format, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()


for file, label in zip(files, labels):
    file = join(filepath, file)
    df = pd.read_csv(file, skiprows=1)
    df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng'}, errors="raise")
    df = df.loc[(df['lat'] != 0)]
    plot_series(df['lng'], df['lat'], xlabel='Longitude', ylabel='Latitude', label=label)
    plt.show()
