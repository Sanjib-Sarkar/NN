import matplotlib.colors as pcolor
import numpy as np
import pandas as pd
import pymap3d as pm
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

tf.keras.backend.clear_session()
plt.style.use('dark_background')
colors = [k for k, v in pcolor.cnames.items()]


def data_cleanup(sub_df):
    """ input: df[['time','lat', 'lng']]"""
    dff = sub_df.resample('1S', on='time').mean()
    dff = dff.interpolate(method='linear')
    return dff


file = 'Data/20220222_153859_hlc.csv'
df = pd.read_csv(file, skiprows=1)

start = 11550  # 11700
end = 14000  # 13900
df = df.iloc[start:end]

# df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng'}, errors="raise")
# df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng', "Speed (kts)": 'speed'},
# errors="raise")
df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng',
                        "Date (mm/dd/yyyy)": 'Date', "Time": 'time'}, errors="raise")

df = df.loc[(df['lat'] != 0)]  # discord 0 values
df.time = pd.to_datetime(df['Date'] + ' ' + df['time']).round('1S')
dff = df[['time', 'lat', 'lng']].copy()
dff = data_cleanup(dff)

dff['h'] = 0
df.reset_index(drop=True, inplace=True)

# df = df.iloc[start:end]
#
# df.reset_index(drop=True, inplace=True)

ori_lat, ori_lng = dff.lat[0], dff.lng[0]
# print(df.head(1), ori_lat, ori_lng)

' Convert to local coordinate system'
dff['e_lat'], dff['n_lng'], dff['u'] = pm.geodetic2enu(dff.lat, dff.lng, dff.h, ori_lat, ori_lng, dff.h, ell=None, deg=True)
# print(df.e_lat)

lat = dff.e_lat
lng = dff.n_lng

coordinates = np.array([[i, j] for i, j in zip(dff.e_lat, dff.n_lng)])


def train_test_split(dataset, percentage):
    train_size = int(len(dataset) * percentage)
    # test_size = len(dataset) - train_size
    train_data, test_data = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    return train_data, test_data


def ts_from_array(features, targets, window, batch, shuffle):
    # data = features[:-(window + 1)] if future != 1 else features
    print("Features and targets shape: ", features.shape, targets.shape if targets is not None else '')
    tensor = tf.keras.preprocessing.timeseries_dataset_from_array(features, targets, sequence_length=window,
                                                                  sequence_stride=1,
                                                                  sampling_rate=1, batch_size=batch, shuffle=shuffle,
                                                                  seed=None, start_index=None, end_index=None)

    return tensor


def scaling(data):
    # mn_scaler = MinMaxScaler(feature_range=(-1, 1))
    mn_scaler = MinMaxScaler()
    mn_scaler.fit(data)
    data = mn_scaler.transform(data)
    return mn_scaler, data


def sequence_data(series, window_past: int, window_future: int) -> np:
    """ series: whole data
        window_past = no. of past observation
        window_future = no of future observation"""
    x, y = list(), list()
    for w_start in range(len(series)):
        past_end = w_start + window_past
        f_end = past_end + window_future
        if f_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[w_start:past_end, :], series[past_end:f_end, ]
        x.append(past)
        y.append(future)
    return np.array(x), np.array(y)


scaler, coordinates = scaling(coordinates)

train, test = train_test_split(coordinates, 0.90)

n_past = 64
n_future = 64
n_features = 2

X_train, y_train = sequence_data(train, n_past, n_future)
X_train = X_train.reshape((-1, X_train.shape[1], n_features))
y_train = y_train.reshape((-1, y_train.shape[1], n_features))

X_test, y_test = sequence_data(test, n_past, n_future)
X_test = X_test.reshape((-1, X_test.shape[1], n_features))
y_test = y_test.reshape((-1, y_test.shape[1], n_features))

print(f'Total:{coordinates.shape}, Train: {train.shape}, Test: {test.shape},'
      f'Features: {X_train.shape}, Targets: {y_train.shape},test-x: {X_test.shape}, test-y: {y_test.shape}')


# n_future = 30/5
def model1(n_past, n_features):
    units = 8
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))

    encoder_outputs1 = tf.keras.layers.LSTM(units, return_state=True)(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]

    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])

    decoder_l1 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation='linear'))(
        decoder_l1)

    model_e1d1 = tf.keras.models.Model(encoder_inputs, decoder_outputs1)

    model_e1d1.summary()
    return model_e1d1


def model_lstm(n_past, n_features):
    units = 128
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, input_shape=(n_past, n_features)),
        tf.keras.layers.RepeatVector(n_past),
        tf.keras.layers.LSTM(units, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation='linear'))])
    model.summary()
    return model


def model2(n_past, n_features):
    # E2D2
    # n_features ==> no of features at each timestep in the data.
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
    units = 128
    encoder_l1 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]

    encoder_l2 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]

    encoder_l3 = tf.keras.layers.LSTM(units, return_state=True)
    encoder_outputs3 = encoder_l3(encoder_outputs2[0])
    encoder_states3 = encoder_outputs3[1:]

    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs3[0])
    #
    decoder_l1 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_l2 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_l1, initial_state=encoder_states2)
    decoder_l3 = tf.keras.layers.LSTM(units, return_sequences=True)(decoder_l2, initial_state=encoder_states3)
    decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation='linear'))(
        decoder_l3)
    #
    model_e2d2 = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
    #
    model_e2d2.summary()

    return model_e2d2


def model1_bi(n_past, n_features):
    units = 128
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))

    encoder_outputs1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_state=True, dropout=0.5,
                                                                          ))(encoder_inputs)

    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])

    decoder_l1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True, dropout=0.5))(
        decoder_inputs, initial_state=encoder_outputs1[1:])
    decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation='linear'))(
        decoder_l1)

    model_e1d1 = tf.keras.models.Model(encoder_inputs, decoder_outputs1)

    model_e1d1.summary()
    return model_e1d1


def model2_bi(n_past, n_features):
    # E2D2
    # n_features ==> no of features at each timestep in the data.
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
    units = 256
    encoder_l1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True, return_state=True))
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]

    encoder_l2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True, return_state=True))
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]

    encoder_l3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_state=True))
    encoder_outputs3 = encoder_l3(encoder_outputs2[0])
    encoder_states3 = encoder_outputs3[1:]

    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs3[0])
    #
    decoder_l1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(decoder_inputs,
                                                                                                   initial_state=encoder_states1)
    decoder_l2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(decoder_l1,
                                                                                                   initial_state=encoder_states2)
    decoder_l3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(decoder_l2,
                                                                                                   initial_state=encoder_states3)
    decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation='linear'))(
        decoder_l3)
    #
    model_e2d2 = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
    #
    model_e2d2.summary()

    return model_e2d2


# callback_reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.80 ** x)
callback_es_train = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
callback_es_val = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

model = model1_bi(n_past, n_features)
# model = model_lstm(n_past, n_features)

lr = 0.0001
optimiser = tf.keras.optimizers.Adam(learning_rate=lr)
# optimiser = tf.keras.optimizers.Adam()
# # # # *************************LearningRate*******************************************************
# lr_schedular = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr * 10 ** (epoch / 20))
# es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
# model.compile(optimizer=optimiser, loss='mse', metrics=['mae'])
# epochs = 100
# history = model.fit(X_train, y_train, epochs=epochs, verbose=1, callbacks=[lr_schedular, es_callback])
# plt.semilogx(history.history["lr"], history.history["loss"], label='loss')
# # plt.axis([1e-3, 1, 0, 0.4])
# plt.xlabel('Learning rate')
# plt.legend()
# plt.show()
# # # # *************************LearningRate*******************************************************

model.compile(optimizer=optimiser, loss='mse', metrics=['mae'])
history_e1d1 = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=n_past, verbose=1,
                         callbacks=[callback_es_train, callback_es_val], shuffle=True)
fig1, ax1 = plt.subplots()
ax1.plot(history_e1d1.history['loss'], 'r', label='TrainLoss')
ax1.plot(history_e1d1.history['val_loss'], 'y', label='ValidationLoss')
plt.legend()
plt.show()

# xtest, ytest = sequence_data(coordinates, n_past, n_future)
# X_test = xtest.reshape((X_test.shape[0], X_test.shape[1], n_features))

# pred = model.predict(X_test)
#
# print(f'Predicted shape: {pred.shape}')
# pred = pred.reshape((-1, n_features))
# print(f'Predicted shape: {pred.shape}')
# pred = scaler.inverse_transform(pred)

# pred_lat, pred_lng, _ = pm.enu2geodetic(pred[:, :1], pred[:, 1:], 0, df.lat[index], df.lng[index], 0, ell=None,
#                                         deg=True)

# plt.plot(pred_lng, pred_lat)

t_start = 1190

taj_lat = dff.e_lat[t_start - n_past: t_start]
taj_lng = dff.n_lng[t_start - n_past: t_start]
print(f'size:lat: {taj_lat.shape}')
t_data = np.array([[i, j] for i, j in zip(taj_lat, taj_lng)])

print(f'shape: pdata{t_data.shape}')

scaler2, coordinates = scaling(t_data)
t_data = np.reshape(coordinates, (1, X_train.shape[1], X_train.shape[2]))

predicted = model.predict(t_data)
predicted = predicted.reshape((-1, n_features))

p_cor = scaler2.inverse_transform(predicted)
p_lat, p_lng = p_cor[:, :1], p_cor[:, 1:]

plt.plot(lng, lat, label='Original', color='r')
plt.scatter(p_lng, p_lat, label=f'predicted future: {n_future}', color='y')
plt.scatter(taj_lng, taj_lat, marker='*', label=f'History given: {n_past}')
plt.legend()
plt.show()

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
