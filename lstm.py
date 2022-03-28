import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from matplotlib import pyplot as plt
import pymap3d as pm

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error

plt.style.use('dark_background')

file = 'Data/20220222_153859_hlc.csv'
df = pd.read_csv(file, skiprows=1)
df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng'}, errors="raise")
df = df.loc[(df['lat'] != 0)]
df['h'] = 0

' Convert to local coordinate system'
df['e_lat'], df['n_lng'], df['u'] = pm.geodetic2enu(df.lat, df.lng, df.h, df.lat[len(df.lat) - 1],
                                                    df.lng[len(df.lng) - 1], df.h, ell=None, deg=True)

start = 10500
end = 13700
series_time = df.Time[start:end].to_list
series_lat = df.e_lat[start:end]
series_lng = df.n_lng[start:end]

coordinates = [[i, j] for i, j in zip(series_lat, series_lng)]
coordinates = np.array(coordinates)


# print(coordinates[:10])

def ts_data_preparation(features, window=5, future=1, batch=1, shuffle=False):
    target = features[(future - 1):]
    features = features[:-(future - 1)] if future != 1 else features
    tensor = TimeseriesGenerator(features, target, length=window, batch_size=batch, shuffle=shuffle)
    return tensor


def test_data_preparation(series, window_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    # dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def train_test_split(dataset, percentage):
    train_size = int(len(dataset) * percentage)
    # test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    return train, test


train_coor, test_coor = train_test_split(coordinates, 0.7)

window_size = 15
future = 1
batch_size = 32

train_coor = ts_data_preparation(train_coor, window=window_size, future=future, batch=batch_size)


def model_lstm(w_size):
    units = 64
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units, activation='relu', return_sequences=True, input_shape=(w_size, 2)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(2, activation='linear')])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                  loss='mse')
    model.summary()
    # print('Train...', model.summary())
    return model


model = model_lstm(window_size)

epochs = 50

model.fit(train_coor, epochs=epochs, verbose=1)



# fig, ax = plt.subplots(1, 1, figsize=(7, 5), tight_layout=True, dpi=100)
# file = 'Data/20220222_153859_hlc.csv'
# df = pd.read_csv(file, skiprows=1)
# df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng'}, errors="raise")
# df = df.loc[(df['lat'] != 0)]
#
# start = 10500
# end = 13700
# series_time = df.Time[start:end].to_numpy()
# series_lat = df.lat[start:end].to_numpy()
# series_lng = df.lng[start:end].to_numpy()
#
#
# def train_test_split(dataset, percentage):
#     train_size = int(len(dataset) * percentage)
#     # test_size = len(dataset) - train_size
#     train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
#     return train, test
#
#
# def test_data_preparation(series, window_size, batch_size):
#     dataset = tf.data.Dataset.from_tensor_slices(series)
#     dataset = dataset.window(window_size, shift=1, drop_remainder=True)
#     dataset = dataset.flat_map(lambda window: window.batch(window_size))
#     # dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
#     dataset = dataset.batch(batch_size).prefetch(1)
#     return dataset
#
#
# def ts_data_preparation(features, window=5, future=1, batch=1, shuffle=False):
#     target = features[(future - 1):]
#     features = features[:-(future - 1)] if future != 1 else features
#     tensor = TimeseriesGenerator(features, target, length=window, batch_size=batch, shuffle=shuffle)
#     return tensor
#
#
# # reshape
# series_lat = np.reshape(series_lat, (len(series_lat), 1))
# series_lng = np.reshape(series_lng, (len(series_lng), 1))
#
# # split into train and test sets
# train_lat, test_lat = train_test_split(series_lat, 0.7)
# train_lng, test_lng = train_test_split(series_lng, 0.7)
#
# tl = test_lat
# tln = test_lng
#
# window_size = 10
# future = 2
# batch_size = 32
#
# train_lat = ts_data_preparation(train_lat, window=window_size, future=future, batch=batch_size,
#                                 shuffle=True)
# train_lng = ts_data_preparation(train_lng, window=window_size, future=future, batch=batch_size,
#                                 shuffle=True)
#
# test_lat = test_data_preparation(test_lat, window_size=window_size, batch_size=batch_size)
# test_lng = test_data_preparation(test_lng, window_size=window_size, batch_size=batch_size)
#
#
# def model_lstm(w_size):
#     units = 50
#     tf.keras.backend.clear_session()
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.LSTM(units, activation='relu', return_sequences=True, input_shape=(w_size, 1)),
#         tf.keras.layers.LSTM(units, activation='relu'),
#         tf.keras.layers.Dense(1, activation='linear')])
#     model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-7, momentum=0.9),
#                   loss='mse')
#     model.summary()
#     # print('Train...', model.summary())
#     return model
#
#
# model = model_lstm(window_size)
#
# epochs = 50
#
# model.fit(train_lat, epochs=epochs, verbose=1)
# testPredict_lat = model.predict(test_lat)
#
# model_ln = model_lstm(window_size)
# model_ln.fit(train_lng, epochs=epochs, verbose=1)
# testPredict_lng = model_ln.predict(test_lng)
#
# plt.plot(testPredict_lng, testPredict_lat, color='y', label=f'P:future:{future}')
# plt.plot(tln, tl, color='r', label='Original')
#
# plt.title(f'WindowSize: {window_size}')
# plt.legend()
# plt.show()
