import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

file = 'Data/20220222_153859_hlc.csv'
df = pd.read_csv(file, skiprows=1)
df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng'}, errors="raise")
# print(df.columns)
# print(df.head(5))
df = df.loc[(df['lat'] != 0)]
# print(len(df.lat))
start = 11400
end = 13700
series_time = df.Time[start:end]
series_lat = df.lat[start:end]
series_lng = df.lng[start:end]

series_time = series_time.to_numpy()
series_lat = series_lat.to_numpy()
series_lng = series_lng.to_numpy()


# # print(type(series_lng))
#
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
# ax1.plot(df['lng'], df['lat'])
# ax2.plot(series_lng, series_lat)
#
# fig2, (axa, axb, axc) = plt.subplots(3, 1, figsize=(10, 6))
# axa.plot(series_time, series_lng)
# axa.set_xticks(axa.get_xticks()[::240])
# axa.set_title('Longitude')
# axb.plot(series_time, series_lat)
# axb.set_xticks(axb.get_xticks()[::240])
# axb.set_title('Latitude')
#
# color = 'red'
# axc.plot(series_time, series_lng, color=color)
# axc.set_ylabel('Longitude', color=color)
# axc.set_xticks(axc.get_xticks()[::240])
#
# axy = axc.twinx()  # instantiate a second axes that shares the same x-axis
# axy.plot(series_time, series_lat)
# axy.set_ylabel('Latitude')
#
#
# split_time = 500
# lng_train = series_lng[:split_time]
# lng_valid = series_lng[split_time:]
#
# lat_train = series_lat[:split_time]
# lat_valid = series_lat[split_time:]
#
# naive_forecast_lng = series_lng[split_time - 1:-1]
# naive_forecast_lat = series_lat[split_time - 1:-1]
#
# fig3, ax = plt.subplots(2, 1, figsize=(10, 6))
# ax[0].plot(lng_valid)
# ax[0].plot(naive_forecast_lng)
# print(keras.metrics.mean_squared_error(lng_valid, naive_forecast_lng).numpy())
# print(keras.metrics.mean_absolute_error(lng_valid, naive_forecast_lng).numpy())
# plt.show()
#
# # naive_forecast = series[split_time - 1:-1]

# fig, ax = plt.subplots(1, 1, figsize=(12, 5))
# ax.plot(series_time, series_lng)
# ax.set_xticks(ax.get_xticks()[::240])
#
# # axy = ax.twinx()  # instantiate a second axes that shares the same x-axis
# # axy.plot(series_time, series_lat, color='b')
# fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5))
# ax2.plot(series_time, series_lat)
# ax2.set_xticks(ax.get_xticks()[::240])


# tensor dataset
# dataset = tf.data.Dataset.from_tensor_slices(series_lat)
# # print(tf.data.Dataset.range(10))
# dataset = dataset.window(15, shift=1, drop_remainder=True)
# dataset = dataset.flat_map(lambda window: window.batch(15))
# dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
# dataset = dataset.shuffle(buffer_size=10)
# dataset = dataset.batch(2).prefetch(1)
# for x, y in dataset:
#     print(x.numpy(), y.numpy())


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


# training and validation
split_time = 800
time_train = series_time[:split_time]
x_train_lat = series_lat[:split_time]
x_train_lng = series_lng[:split_time]

time_valid = series_time[split_time:]
x_valid_lat = series_lat[split_time:]
x_valid_lng = series_lng[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset_lat = windowed_dataset(x_train_lat, window_size=window_size, batch_size=batch_size,
                               shuffle_buffer=shuffle_buffer_size)

dataset_lng = windowed_dataset(x_train_lng, window_size=window_size, batch_size=batch_size,
                               shuffle_buffer=shuffle_buffer_size)

l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([l0])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))
model.fit(dataset_lat, epochs=120, verbose=0)

# print("Layer weights{}".format(l0.get_weights()))

# print(series_lat[1:21])
# print(model.predict(series_lat[1:21][np.newaxis]))

forecast = []
for time in range(len(series_lat) - window_size):
    forecast.append(model.predict(series_lat[time:time + window_size][np.newaxis]))

forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]

fig, ax = plt.subplots(1, 1, figsize=(12, 5))

ax.plot(x_valid_lat)
ax.plot(results)
# ax.set_xticks(ax.get_xticks()[::180])

plt.show()
