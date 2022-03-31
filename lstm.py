import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from matplotlib import pyplot as plt
import pymap3d as pm
import matplotlib.colors as pltc

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

plt.style.use('dark_background')
colors = [k for k, v in pltc.cnames.items()]

file = 'Data/20220222_153859_hlc.csv'
df = pd.read_csv(file, skiprows=1)
df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng'}, errors="raise")
df = df.loc[(df['lat'] != 0)]
df['h'] = 0

index = int(len(df.lat) / 2)

' Convert to local coordinate system'
df['e_lat'], df['n_lng'], df['u'] = pm.geodetic2enu(df.lat, df.lng, df.h, df.lat[index],
                                                    df.lng[index], df.h, ell=None, deg=True)

start = 10600
end = 13700
lat = df.lat[start:end]
lng = df.lng[start:end]

series_time = df.Time[start:end].to_list
series_lat = df.e_lat[start:end]
series_lng = df.n_lng[start:end]

coordinates = [[i, j] for i, j in zip(series_lat, series_lng)]
coordinates = np.array(coordinates)


# print(coordinates[:10])

def train_test_split(dataset, percentage):
    train_size = int(len(dataset) * percentage)
    # test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    return train, test


def ts_from_array(data, target, window, future, batch, shuffle):
    data = data[:-(window + 1)] if future != 1 else data
    targets = target
    print("length: ", len(data), len(targets) if targets is not None else '')
    tensor = tf.keras.preprocessing.timeseries_dataset_from_array(data, targets, sequence_length=window,
                                                                  sequence_stride=1,
                                                                  sampling_rate=1, batch_size=batch, shuffle=shuffle,
                                                                  seed=None,
                                                                  start_index=None, end_index=None)
    return tensor


# create scaler
scaler = StandardScaler()
# fit scaler on data
scaler.fit(coordinates)
# apply transform
coordinates = scaler.transform(coordinates)
# print(coordinates[:17])

train, test = train_test_split(coordinates, 0.8)

window_size = 20
future = 5
batch_size = 32

target = train[(window_size + future) - 1:]

print(f'train: {len(train)}, target: {len(target)}')

xy_train = ts_from_array(train, target, window=window_size, future=future, batch=batch_size, shuffle=True)


# xy_test = ts_from_array(test, None, window=window_size, future=future, batch=batch_size, shuffle=False)


def model_lstm(w_size):
    units = 32
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM((w_size+2)*2, activation='relu', return_sequences=True, input_shape=(w_size, 2)),
        # tf.keras.layers.LSTM(units, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(units, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(2, activation='linear')])
    model.summary()
    return model


model = model_lstm(window_size)

# lr = 2e-4
# lr_schedular = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr * 10 ** (epoch / 30))
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9), loss='mse', metrics=['mae'])
# epochs = 100
# history = model.fit(xy_train, epochs=epochs, verbose=1, callbacks=[lr_schedular])
# plt.semilogx(history.history["lr"], history.history["loss"], label='loss')
# plt.axis([1e-3, 1, 0, 0.4])
# plt.xlabel('Learning rate')
# plt.legend()
# plt.show()

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=5e-2, momentum=0.9), loss='mse', metrics=['accuracy'])
epochs = 100
target_validation = test[(window_size + future) - 1:]
xy_validation = ts_from_array(test, target_validation, window=window_size, future=future, batch=batch_size,
                              shuffle=False)
history = model.fit(xy_train, epochs=epochs, validation_data=xy_validation, verbose=1)

print(history.history.keys())
# # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
fig1, ax1 = plt.subplots()
ax1.plot(history.history['loss'], 'r', label='TrainLoss')
ax1.plot(history.history['accuracy'], 'orange', label='TrainAccuracy')
ax1.plot(history.history['val_loss'], 'w', label='ValidationLoss')
ax1.plot(history.history['val_accuracy'], 'g', label='ValidationAccuracy')
ax1.set_xlabel('Epochs')
# plt.grid(visible=True, axis='both')
ax1.set_title(f'Window:{window_size}, future: {future}')
ax1.legend()
# plt.show()

test = ts_from_array(coordinates, None, window=window_size, future=future, batch=batch_size, shuffle=False)

predicted = model.predict(test)
predicted_coor = scaler.inverse_transform(predicted)

df_n = pd.DataFrame(predicted_coor, columns=('p_lat', 'p_lng'))
df_n['h'] = 0
' Back to lag lng'
df_n['Predted_lat'], df_n['Predicted_lng'], _ = pm.enu2geodetic(df_n.p_lat, df_n.p_lng, df_n.h,
                                                                df.lat[index], df.lng[index],
                                                                df.h[index], ell=None, deg=True)
fig, ax = plt.subplots()
ax.plot(df_n.Predicted_lng, df_n.Predted_lat, color='r', label=f'predicted, future:{future}')
ax.plot(lng, lat, label='Original')
ax.set_title(f'Window{window_size}, future:{future}')
ax.legend()
plt.show()


###  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
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
