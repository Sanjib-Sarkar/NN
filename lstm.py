import matplotlib.colors as pcolor
import numpy as np
import pandas
import pandas as pd
import pymap3d as pm
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

plt.style.use('dark_background')
font = {'family': 'Arial', 'weight': 'normal', 'size': 10}
plt.rc('font', **font)
colors = [k for k, v in pcolor.cnames.items()]

# file = 'Data/simulated-data.csv'
# df = pd.read_csv(file, skiprows=0)
# start = 10
# end = 6400

file = 'Data/20220222_153859_hlc.csv'
df = pd.read_csv(file, skiprows=1)
df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng'}, errors="raise")
# df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng', "Speed (kts)": 'speed'},
# errors="raise")
start = 11700
end = 13900

df = df.loc[(df['lat'] != 0)]  # discord 0 values
df['h'] = 0
df.reset_index(drop=True, inplace=True)

df = df.iloc[start:end]

df.reset_index(drop=True, inplace=True)

# ori_lat, ori_lng = df.lat[0], df.lng[0]
ori_lat, ori_lng = df.lat[len(df.lat) - 1], df.lng[len(df.lng) - 1]

' Convert to local coordinate system'
df['e_lat'], df['n_lng'], df['u'] = pm.geodetic2enu(df.lat, df.lng, df.h, ori_lat, ori_lng, df.h, ell=None, deg=True)

sub_df = df[['e_lat', 'n_lng']].copy()


def scaling(data: pandas.DataFrame):
    # mn_scaler = MinMaxScaler(feature_range=(-1, 1))
    data = data.copy()
    mn_scaler = MinMaxScaler()
    mn_scaler.fit(data)
    data = mn_scaler.transform(data)
    data = pd.DataFrame(data, columns=(['lat', 'lng']))
    return mn_scaler, data


def sequence_data(df: pd.DataFrame, window_size, forecast_size, batch_size):
    shuffle_buffer_size = len(df)
    # Total size of window is given by the number of steps to be considered
    # before prediction time + steps that we want to forecast
    total_size = window_size + forecast_size

    data = tf.data.Dataset.from_tensor_slices(df.values)

    # Selecting windows
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    # Shuffling data (seed=Answer to the Ultimate Question of Life, the Universe, and Everything)
    data = data.shuffle(shuffle_buffer_size)

    # Extracting past features + deterministic future + labels
    data = data.map(lambda k: ((k[:-forecast_size],
                                k[-forecast_size:, :])))

    return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


n_past = 60
n_future = 30
n_features = 2
mn_scaler, scaled_data = scaling(sub_df)
# print(type(scaled_data))
shuffled_data = sequence_data(scaled_data, window_size=n_past, forecast_size=n_future, batch_size=32)

total_size = len(list(shuffled_data.as_numpy_iterator()))

percentage = 0.8
train_size = int(total_size * percentage)
test_size = 1 - train_size

train = shuffled_data.take(train_size)
test = shuffled_data.take(test_size)


def model1(n_past, n_features):
    units = 200
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))

    # encoder_l1 = tf.keras.layers.LSTM(units, return_state=True)
    # encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_outputs1 = tf.keras.layers.LSTM(units, return_state=True)(encoder_inputs)
    # check = np.array(encoder_outputs1)
    # print('len:', len(encoder_outputs1), encoder_outputs1[1:])
    encoder_states1 = encoder_outputs1[1:]

    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])

    decoder_l1 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation='linear'))(
        decoder_l1)

    model_e1d1 = tf.keras.models.Model(encoder_inputs, decoder_outputs1)

    model_e1d1.summary()
    return model_e1d1


def model2(n_past, n_features):
    # E2D2: encoder_outputs, state_h, state_c
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
    units = 256
    encoder_outputs1 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]

    encoder_outputs2 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]

    encoder_outputs3 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)(encoder_outputs2[0])
    encoder_states3 = encoder_outputs3[1:]

    encoder_outputs4 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)(encoder_outputs3[0])
    encoder_states4 = encoder_outputs4[1:]

    encoder_outputs5 = tf.keras.layers.LSTM(units, return_state=True)(encoder_outputs4[0])
    encoder_states5 = encoder_outputs5[1:]

    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs5[0])
    #
    decoder_l1 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_l2 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_l1, initial_state=encoder_states2)
    decoder_l3 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_l2, initial_state=encoder_states3)
    decoder_l4 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_l3, initial_state=encoder_states4)
    decoder_l5 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_l4, initial_state=encoder_states5)
    decoder_outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation='linear'))(
        decoder_l5)
    #
    model_e2d2 = tf.keras.models.Model(encoder_inputs, decoder_outputs)
    #
    model_e2d2.summary()

    return model_e2d2


model = model2(n_past=n_past, n_features=n_features)

callback_es_train = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
callback_es_val = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
lr = 0.00001
optimiser = tf.keras.optimizers.Adam(learning_rate=lr)
# # # *************************LearningRate*******************************************************
# lr_schedular = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr * 10 ** (epoch / 20))
# es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
# model.compile(optimizer=optimiser, loss='mse', metrics=['mae'])
# epochs = 100
# history = model.fit(train, epochs=epochs, verbose=1, callbacks=[lr_schedular, es_callback])
# plt.semilogx(history.history["lr"], history.history["loss"], label='loss')
# # plt.axis([1e-3, 1, 0, 0.4])
# plt.xlabel('Learning rate')
# plt.legend()
# plt.show()
# # # *************************LearningRate*******************************************************

model.compile(optimizer=optimiser, loss='mse', metrics=['mae'])
history_model = model.fit(train, epochs=100, validation_data=test, verbose=1,
                          callbacks=[callback_es_train, callback_es_val])
fig1, ax1 = plt.subplots()
ax1.plot(history_model.history['loss'], 'r', label='TrainLoss')
ax1.plot(history_model.history['val_loss'], 'y', label='ValidationLoss')
ax1.legend()
# plt.show()


# plt.show()

# start_p = 5500
start_p = 1150
x_test = sub_df[start_p: start_p + n_past]
# print(x_test)
x_scaler, x_scaled_test = scaling(x_test)
x_scaled_test = np.array(x_scaled_test)
x_scaled_test = x_scaled_test.reshape((1, -1, n_features))

# print(type(x_test), x_test[:5])
pred = model.predict(x_scaled_test)
pred = np.squeeze(pred, axis=0)

df_pred = pd.DataFrame(pred, columns=(['lat', 'lng']))

df_pred = x_scaler.inverse_transform(df_pred)
df_pred = pd.DataFrame(df_pred, columns=(['lat', 'lng']))
# print(type(df_pred), df_pred)
fig2, ax2 = plt.subplots()

ax2.plot(sub_df.n_lng, sub_df.e_lat, color='r')
ax2.plot(df_pred.lng, df_pred.lat, 'r*', label=f'predicted:{n_future}')
ax2.plot(x_test.n_lng, x_test.e_lat, 'yo', label=f'past data:{n_past}')
ax2.legend()
plt.show()

# print(df_pred.head(5))

# lat = df.e_lat
# lng = df.n_lng
#
# coordinates = np.array([[i, j] for i, j in zip(df.e_lat, df.n_lng)])
#
#
# def train_test_split(dataset, percentage):
#     train_size = int(len(dataset) * percentage)
#     # test_size = len(dataset) - train_size
#     train_data, test_data = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
#     return train_data, test_data
#
#
# def ts_from_array(features, targets, window, batch, shuffle):
#     # data = features[:-(window + 1)] if future != 1 else features
#     print("Features and targets shape: ", features.shape, targets.shape if targets is not None else '')
#     tensor = tf.keras.preprocessing.timeseries_dataset_from_array(features, targets, sequence_length=window,
#                                                                   sequence_stride=1,
#                                                                   sampling_rate=1, batch_size=batch, shuffle=shuffle,
#                                                                   seed=None, start_index=None, end_index=None)
#
#     return tensor
#
#
# def scaling(data):
#     # mn_scaler = MinMaxScaler(feature_range=(-1, 1))
#     mn_scaler = MinMaxScaler()
#     mn_scaler.fit(data)
#     data = mn_scaler.transform(data)
#     return mn_scaler, data
#
#
# def sequence_data(series, window_past: int, window_future: int) -> np:
#     """ series: whole data
#         window_past = no. of past observation
#         window_future = no of future observation"""
#     x, y = list(), list()
#     for w_start in range(len(series)):
#         past_end = w_start + window_past
#         f_end = past_end + window_future
#         if f_end > len(series):
#             break
#         # slicing the past and future parts of the window
#         past, future = series[w_start:past_end, :], series[past_end:f_end, :]
#         x.append(past)
#         y.append(future)
#     return np.array(x), np.array(y)
#
#
# scaler, coordinates = scaling(coordinates)
#
# n_past = 60
# n_future = 30
# n_features = 2
#
# X_train, y_train = sequence_data(coordinates, n_past, n_future)
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
# y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
#
# X_test, y_test = sequence_data(test, n_past, n_future)
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
# y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
#
# train, test = train_test_split(coordinates, 0.80)
#
# print(f'Total:{coordinates.shape}, Train: {train.shape}, Test: {test.shape},'
#       f'Features: {X_train.shape}, Targets: {y_train.shape},test-x: {X_test.shape}, test-y: {y_test.shape}')
#
#
# def model1(n_past, n_features):
#     units = 200
#     encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
#
#     # encoder_l1 = tf.keras.layers.LSTM(units, return_state=True)
#     # encoder_outputs1 = encoder_l1(encoder_inputs)
#     encoder_outputs1 = tf.keras.layers.LSTM(units, return_state=True)(encoder_inputs)
#     # check = np.array(encoder_outputs1)
#     # print('len:', len(encoder_outputs1), encoder_outputs1[1:])
#     encoder_states1 = encoder_outputs1[1:]
#
#     decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])
#
#     decoder_l1 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
#     decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation='linear'))(
#         decoder_l1)
#
#     model_e1d1 = tf.keras.models.Model(encoder_inputs, decoder_outputs1)
#
#     model_e1d1.summary()
#     return model_e1d1
#
#
# def model_lstm(n_past, n_features):
#     units = 32
#     tf.keras.backend.clear_session()
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, input_shape=(n_past, n_features)),
#         tf.keras.layers.RepeatVector(n_past),
#         tf.keras.layers.LSTM(units, return_sequences=True),
#         tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation='linear'))])
#     model.summary()
#     return model
#
#
# def model2(n_past, n_features):
#     # E2D2
#     # n_features ==> no of features at each timestep in the data.
#     encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
#     units = 200
#     encoder_l1 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
#     encoder_outputs1 = encoder_l1(encoder_inputs)
#     encoder_states1 = encoder_outputs1[1:]
#
#     encoder_l2 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
#     encoder_outputs2 = encoder_l2(encoder_outputs1[0])
#     encoder_states2 = encoder_outputs2[1:]
#
#     encoder_l3 = tf.keras.layers.LSTM(units, return_state=True)
#     encoder_outputs3 = encoder_l3(encoder_outputs2[0])
#     encoder_states3 = encoder_outputs3[1:]
#
#     decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs3[0])
#     #
#     decoder_l1 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
#     decoder_l2 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_l1, initial_state=encoder_states2)
#     decoder_l3 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_l2, initial_state=encoder_states3)
#     decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation='linear'))(
#         decoder_l3)
#     #
#     model_e2d2 = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
#     #
#     model_e2d2.summary()
#
#     return model_e2d2
#
#
# # callback_reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.80 ** x)
# callback_es_train = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
# callback_es_val = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
#
# model = model1(n_past, n_features)
# # model = model_lstm(n_past, n_features)
#
# lr = 0.00001
# optimiser = tf.keras.optimizers.Adam(learning_rate=lr)
# # optimiser = tf.keras.optimizers.Adam()
# # # # *************************LearningRate*******************************************************
# # lr_schedular = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr * 10 ** (epoch / 20))
# # es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
# # model.compile(optimizer=optimiser, loss='mse', metrics=['mae'])
# # epochs = 100
# # history = model.fit(X_train, y_train, epochs=epochs, verbose=1, callbacks=[lr_schedular, es_callback])
# # plt.semilogx(history.history["lr"], history.history["loss"], label='loss')
# # # plt.axis([1e-3, 1, 0, 0.4])
# # plt.xlabel('Learning rate')
# # plt.legend()
# # plt.show()
# # # # *************************LearningRate*******************************************************
#
# model.compile(optimizer=optimiser, loss='mse', metrics=['mae'])
# history_e1d1 = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32, verbose=1,
#                          callbacks=[callback_es_train, callback_es_val])
# fig1, ax1 = plt.subplots()
# ax1.plot(history_e1d1.history['loss'], 'r', label='TrainLoss')
# ax1.plot(history_e1d1.history['val_loss'], 'y', label='ValidationLoss')
# plt.legend()
# plt.show()
#
# # xtest, ytest = sequence_data(coordinates, n_past, n_future)
# # X_test = xtest.reshape((X_test.shape[0], X_test.shape[1], n_features))
#
# # pred = model.predict(X_test)
# #
# # print(f'Predicted shape: {pred.shape}')
# # pred = pred.reshape((-1, n_features))
# # print(f'Predicted shape: {pred.shape}')
# # pred = scaler.inverse_transform(pred)
#
# # pred_lat, pred_lng, _ = pm.enu2geodetic(pred[:, :1], pred[:, 1:], 0, df.lat[index], df.lng[index], 0, ell=None,
# #                                         deg=True)
#
# # plt.plot(pred_lng, pred_lat)
#
# t_start = 1500
#
# taj_lat = df.e_lat[t_start - n_past: t_start]
# taj_lng = df.n_lng[t_start - n_past: t_start]
# print(f'size:lat: {taj_lat.shape}')
# t_data = np.array([[i, j] for i, j in zip(taj_lat, taj_lng)])
#
# print(f'shape: pdata{t_data.shape}')
#
# scaler2, coordinates = scaling(t_data)
# t_data = np.reshape(coordinates, (1, X_train.shape[1], X_train.shape[2]))
#
# predicted = model.predict(t_data)
# predicted = predicted.reshape((-1, n_features))
#
# p_cor = scaler2.inverse_transform(predicted)
# p_lat, p_lng = p_cor[:, :1], p_cor[:, 1:]
#
# plt.plot(lng, lat, label='Original', color='r')
# plt.scatter(p_lng, p_lat, label=f'predicted future: {n_future}', color='y')
# plt.scatter(taj_lng, taj_lat, marker='*', label=f'History given: {n_past}')
# plt.legend()
# plt.show()

# #*************************************************************************************************************8
# print(taj, p_lat, p_lng)

# xy_train = ts_from_array(features, target, window=window_size, batch=batch_size, shuffle=True)

# print("TimeSeries: ", list(xy_train.as_numpy_iterator())[:2])

# def model_lstm(w_size):
#     units = 32
#     tf.keras.backend.clear_session()
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.LSTM(w_size * 2, return_sequences=True, input_shape=(w_size, 4, 2)),
#         tf.keras.layers.LSTM(units),
#         tf.keras.layers.Dense(4, activation='linear')])
#     model.summary()
#     return model
#
#
# model = model_lstm(window_size)
# lr = 0.001
# # optimiser = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
# optimiser = tf.keras.optimizers.Adam(learning_rate=lr)
#
# # lr_schedular = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr * 10 ** (epoch / 25))
# # es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
# # model.compile(optimizer=optimiser, loss='mse', metrics=['mae'])
# # epochs = 100
# # history = model.fit(xy_train, epochs=epochs, verbose=1, callbacks=[lr_schedular, es_callback])
# # plt.semilogx(history.history["lr"], history.history["loss"], label='loss')
# # # plt.axis([1e-3, 1, 0, 0.4])
# # plt.xlabel('Learning rate')
# # plt.legend()
# # plt.show()
#
# model.compile(optimizer=optimiser, loss='mse', metrics=['accuracy'])
# epochs = 100
#
# # valid_target = test[(window_size + future):]
# valid_features = test[:-(window_size + future)]
#
# t1 = test[(window_size + future):]
# t2 = test[(window_size + future) + 1:]
# t3 = test[(window_size + future) + 2:]
# t4 = test[(window_size + future) + 3:]
# # t5 = test[(window_size + future)+4:]
#
# valid_target = np.array([[i, j, k, l] for i, j, k, l in zip(t1, t2, t3, t4)])
# xy_validation = ts_from_array(valid_features, valid_target, window=window_size, batch=batch_size, shuffle=False)
#
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# history = model.fit(xy_train, epochs=epochs, validation_data=xy_validation, verbose=1, callbacks=callback)
#
# # print(history.history.keys())
# # # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
# fig1, ax1 = plt.subplots()
# ax1.plot(history.history['loss'], 'r', label='TrainLoss')
# # ax1.plot(history.history['accuracy'], 'orange', label='TrainAccuracy')
# ax1.plot(history.history['val_loss'], 'w', label='ValidationLoss', zorder=1)
# # ax1.plot(history.history['val_accuracy'], 'g', label='ValidationAccuracy')
# ax1.set_xlabel('Epochs')
# # plt.grid(visible=True, axis='both')
# ax1.set_title(f'Window:{window_size}, future: {future}')
# ax1.legend()
# # plt.show()
#
# test = ts_from_array(test, None, window=window_size, batch=1, shuffle=False)
#
# predicted = model.predict(test)
# # print(f'Predicted:{predicted}: type{type(predicted)}')
# df_n = pd.DataFrame(predicted, columns=('e', 'n'))
# predicted_coor = scaler.inverse_transform(df_n)
#
# df_n = pd.DataFrame(predicted_coor, columns=('p_lat', 'p_lng'))
# df_n['h'] = 0
# ' Back to lag lng'
# df_n['Predted_lat'], df_n['Predicted_lng'], _ = pm.enu2geodetic(df_n.p_lat, df_n.p_lng, df_n.h,
#                                                                 df.lat[index], df.lng[index],
#                                                                 df.h[index], ell=None, deg=True)
# fig, ax = plt.subplots()
# ax.plot(df_n.Predicted_lng, df_n.Predted_lat, color='r', label=f'predicted, future:{future}')
# ax.plot(lng, lat, label='Original')
# ax.set_title(f'Window{window_size}, future:{future}')
# ax.legend()
# # plt.show()
