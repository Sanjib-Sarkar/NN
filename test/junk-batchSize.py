import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import numpy as np

file = 'Data/20220222_153859_hlc.csv'
df = pd.read_csv(file, skiprows=1)
df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng', "Speed (kts)": 'speed'}, errors="raise")
df = df.loc[(df['lat'] != 0)]
df['t'] = [i for i in range(0, len(df.lat))]


# print(df.head(5))


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def ts_data_preparation(features, window_size=5, future=1, batch_size=1, shuffle=False):
    target = features[(future - 1):]
    features = features[:-(future - 1)] if future != 1 else features
    tensor = TimeseriesGenerator(features, target, length=window_size, batch_size=batch_size, shuffle=shuffle)
    x = np.array([i for i, j in tensor], dtype=object)
    y = np.array([j for i, j in tensor], dtype=object)
    return x, y


def ts_from_array(features, targets, window, future, batch, shuffle):
    # data = features[:-(window + 1)] if future != 1 else features
    data = features
    # targets = target
    # print("Features and targets shape: ", data.shape, targets.shape if targets is not None else '')
    tensor = tf.keras.preprocessing.timeseries_dataset_from_array(data, targets, sequence_length=window,
                                                                  sequence_stride=1,
                                                                  sampling_rate=1, batch_size=batch, shuffle=shuffle,
                                                                  seed=None,
                                                                  start_index=None, end_index=None)
    return tensor


# window_size = 5
# future = 1
# batch_size = 1
#
# start = 10500
# end = 13700
#
# x = np.array([[i, j, k] for i, j, k in zip(df.lat, df.lng, df.speed)])
# x = x[start:end]
# y = np.array([[i, j] for i, j in zip(df.lat, df.lng)])
# y = y[start:end]
#
#
# # target = x[(window_size + future) - 1:]
# x = x[:-(future+window_size)]
# target = y[(window_size + future):]
# print(f'train: {x.shape}, target: {target.shape}')
#
# train = ts_from_array(x, target, window=window_size,  future=future, batch=batch_size, shuffle=False)
# # test = ts_from_array(x, None, window=window_size, batch=batch_size, shuffle=False, future=future)
# # print("Coordinates: ", x[:10])
# print("TimeSeries: ", list(train.as_numpy_iterator())[:2])
# # ************************************************************************************************************

def time_series_data_generate(features, labels, ):

    pass

m = [i for i in range(1000)]
n = [j for j in range(1000)]
x = [[i, j] for i, j in zip(m, n)]

window_size = 5
future = 1
batch_size = 1

t1 = x[(window_size + future) - 1:]
t2 = x[(window_size + future+1) - 1:]
# t3 = x[(window_size + future+2) - 1:]
# t4 = x[(window_size + future+3) - 1:]
target = [[i, j] for i, j in zip(t1, t2)]

train = ts_from_array(x, target, window=window_size,  future=future, batch=batch_size, shuffle=False)
print("Coordinates: ", x[:20])
print("TimeSeries: ", list(train.as_numpy_iterator())[:2])


# #********************************************************************************************************

# print("TimeSeries: ", list(test.as_numpy_iterator())[:2])
# wd = windowed_dataset(x, window_size=window_size, batch_size=batch_size,
#                       shuffle_buffer=1)
#
# xtrain, ytrain = ts_data_preparation(x, window_size=window_size, future=future, batch_size=batch_size,
#                                      shuffle=False)

# tensor = tf.keras.utils.timeseries_dataset_from_array(x, targets=None, sequence_length=window_size,
#                                                       batch_size=batch_size, shuffle=False)


# print("TimeSeries: ", train.shape)

# series_lat = df.lat[start:end].to_numpy()
# train_ts_lat = ts_data_preparation(series_lat, window_size=window_size, future=future, batch_size=batch_size,
#                                    shuffle=False)
#
# wd = windowed_dataset(series_lat, window_size=window_size, batch_size=batch_size,
#                       shuffle_buffer=1)
#
# print("series_lat: ", series_lat[:10], len(series_lat))
# print(" train_x_lat: ", train_ts_lat[0])
# # print('Size of wd', len(wd))
# i = 0
# for example in wd:
#     print('X' * 100)
#     print(example[0].numpy())
#     print(example[1].numpy())
#     i += 1
#     if i == 3:
#         break
#
# print(i)
