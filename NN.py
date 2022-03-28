import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tensorflow import keras
import tensorflow as tf
from keras import optimizers

# from tensorflow.python.keras import optimizers

fig, ax = plt.subplots(1, 1, figsize=(12, 5), tight_layout=True, dpi=100)
file = 'Data/20220222_153859_hlc.csv'
df = pd.read_csv(file, skiprows=1)
df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng'}, errors="raise")
# print(df.columns)
# print(df.head(5))
df = df.loc[(df['lat'] != 0)]
# print(len(df.lat))
start = 8500
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
split_time = 3900
time_train = series_time[:split_time]
x_train_lat = series_lat[:split_time]
x_train_lng = series_lng[:split_time]

time_valid = series_time[split_time:]
x_valid_lat = series_lat[split_time:]
x_valid_lng = series_lng[split_time:]
print(f'total: {len(series_lat)}, x_train_lat:{len(x_train_lat)}, x_valid_lat:{len(x_valid_lat)}')

window_size = 15
batch_size = 32
shuffle_buffer_size = 100

dataset_lat = windowed_dataset(x_train_lat, window_size=window_size, batch_size=batch_size,
                               shuffle_buffer=shuffle_buffer_size)

dataset_lng = windowed_dataset(x_train_lng, window_size=window_size, batch_size=batch_size,
                               shuffle_buffer=shuffle_buffer_size)
# print(dataset_lng)
optimisers = [tf.keras.optimizers.SGD]
# tf.keras.optimizers.SGD, tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop,
# tf.keras.optimizers.Adadelta, tf.keras.optimizers.Adagrad, tf.keras.optimizers.Adamax,
# tf.keras.optimizers.Nadam, tf.keras.optimizers.Ftrl]
labels = ['SGD']

# l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([tf.keras.layers.Dense(10,  input_shape=[window_size],  activation="relu"),
                                    tf.keras.layers.Dense(10, activation="relu"),
                                    tf.keras.layers.Dense(1)])
model1 = model

# model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9))
# model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))
# model.fit(dataset_lat, epochs=120, verbose=0)

lr = 1e-6
epochs = 100
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr * 10 ** (epoch / 20))
for optimiser, label in zip(optimisers, labels):
    model.compile(loss="mse", optimizer=optimiser(learning_rate=lr, momentum=0.9))
    model1.compile(loss="mse", optimizer=optimiser(learning_rate=lr, momentum=0.9))

    model.fit(dataset_lat, epochs=epochs, verbose=0)

    # plt.plot(history.history['loss'], label='train')
    # plt.legend()
    # plt.show()

    forecast_lat = []
    for time in range(len(series_lat) - window_size):
        forecast_lat.append(model.predict(series_lat[time:time + window_size][np.newaxis]))
    forecast_lat = forecast_lat[split_time - window_size:]
    results_lat = np.array(forecast_lat)[:, 0, 0]

    model1.fit(dataset_lng, epochs=epochs, verbose=0)

    # Prediction
    forcast_lng = []
    for time in range(len(series_lng) - window_size):
        forcast_lng.append(model1.predict(series_lng[time:time + window_size][np.newaxis]))
    forcast_lng = forcast_lng[split_time - window_size:]
    results_lng = np.array(forcast_lng)[:, 0, 0]

    # Plot prediction
    ax.plot(forcast_lng, forecast_lat, label=label)

    # # plot learning rates vs loss ; Add 'callbacks=[lr_schedule],' inside model.fit
    # lrs = lr * (10 ** (np.arange(epochs) / 20))
    # ax.semilogx(lrs, history.history['loss'])
    # plt.axis([lr, 1e-1, 0, 250])
    # plt.xlabel('Learning Rate')
    # plt.ylabel('Loss')
    # plt.show()

    # # # plot epochs vs loss
    # loss = history.history['loss']
    # epochs = range(3, len(loss))
    # ax.plot(epochs, loss[3:], label='Training Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Error')
    # plt.show()

    # print("Layer weights{}".format(l0.get_weights()))

    # print(series_lat[1:21])
    # print(model.predict(series_lat[1:21][np.newaxis]))

    # forecast = []
    # for time in range(len(series_lat) - window_size):
    #     forecast.append(model.predict(series_lat[time:time + window_size][np.newaxis]))
    #
    # forecast = forecast[split_time - window_size:]
    # results = np.array(forecast)[:, 0, 0]


# print("MSE: ", tf.keras.metrics.mean_absolute_error(x_valid_lat, results).numpy())
# ax.plot(x_valid_lat)
# ax.plot(results)
# plt.scatter(series_time, results)
# ax.set_xticks(ax.get_xticks()[::180])

ax.plot(x_valid_lng, x_valid_lat, label='Latitude')
plt.legend()
plt.show()
