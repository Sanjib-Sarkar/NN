import datetime

import time
import matplotlib.colors as pcolor
import numpy as np
import pandas
import pandas as pd
import pymap3d as pm
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from os.path import exists
import os
import glob
from matplotlib.animation import FuncAnimation

print("Time Start:", datetime.datetime.now())
tf.keras.backend.clear_session()

plt.style.use('dark_background')
font = {'family': 'Arial', 'weight': 'normal', 'size': 10}
plt.rc('font', **font)
colors = [k for k, v in pcolor.cnames.items()]


def preprocessing(data_frame, scaler):
    data_frame.rename(columns={"Latitude": 'lat', "Longitude": 'lng'}, errors="raise", inplace=True)
    data_frame = data_frame[['lat', 'lng', 'Time', 'Date']].copy()
    data_frame = data_frame.loc[(data_frame['lat'] != 0)]
    data_frame.reset_index(drop=True, inplace=True)
    data_frame['time'] = pd.to_datetime(data_frame['Date'] + ' ' + data_frame['Time'])
    data_frame['Time_diff'] = pd.to_timedelta(data_frame['Time'].astype(str)).diff(1).dt.total_seconds()
    data_frame['dt'] = data_frame['Time_diff'].cumsum().fillna(0)
    ori_lat, ori_lng = data_frame.lat[len(data_frame.lat) - 1], data_frame.lng[len(data_frame.lng) - 1]
    data_frame['h'] = 0
    ' Convert to local coordinate system'
    data_frame['e_lat'], data_frame['n_lng'], data_frame['u'] = pm.geodetic2enu(data_frame.lat, data_frame.lng,
                                                                                data_frame.h, ori_lat, ori_lng,
                                                                                data_frame.h, ell=None, deg=True)
    data_frame = data_frame[['e_lat', 'n_lng', 'dt']].copy()

    'Scaling..... ????'
    scaler.fit(data_frame)
    data_frame = scaler.transform(data_frame)
    data_frame = pd.DataFrame(data_frame, columns=(['e_lat', 'n_lng', 'dt']))
    return data_frame


def data_preparation(dframe, past_data_size, forecast_size):
    """sequence data"""
    shuffle_buffer_size = len(dframe)
    total_size = past_data_size + forecast_size
    data = tf.data.Dataset.from_tensor_slices(dframe.values)
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))
    data = data.shuffle(shuffle_buffer_size)
    data = data.map(lambda k: ((k[:-forecast_size, ],
                                k[-forecast_size:, ])))
    return data


# def scaling(data: pandas.DataFrame):  # BiB: changed data -> data1, probably unnecessary
#     mn_scaler = MinMaxScaler(feature_range=(-1., 1.))  # BiB: changed range
#     mn_scaler.fit(data)
#     data1 = mn_scaler.transform(data)
#     data1 = pd.DataFrame(data1, columns=(['lat', 'lng', 'dt']))
#     return mn_scaler, data1, mn_scaler.get_params


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
    data = data.map(lambda k: ((k[:-forecast_size, ],
                                k[-forecast_size:, ])))

    return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


n_past = 60
n_future = 60
n_features = 2 + 1
batch_size = 32
# mn_scaler, scaled_data, parameters = scaling(sub_df)

files = glob.glob("Data/logs/*.log")[:-1]

mn_scaler = MinMaxScaler(feature_range=(-1., 1.))
df = [(data_preparation(preprocessing(pd.read_csv(file, delimiter=';', skiprows=0), mn_scaler), n_past, n_future)) for
      file in files]
# df = tf.data.Dataset.zip((df[0], df[1]))
# df = tf.concat(df, 0)
df = df[0].concatenate(df[1]).concatenate(df[2]).concatenate(df[3]).concatenate(df[4])
df = df.shuffle(len(list(df.as_numpy_iterator())))
df = df.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

total_size = len(list(df.as_numpy_iterator()))

percentage = 0.90
train_size = int(total_size * percentage)
# test_size = 1 - train_size #BiB: suppressed size that was using all data for test anyway

train = df.take(train_size)
test = df.shuffle(total_size).take(-1)  # BiB: using all data for test, as before


def model1_bi(n_past, n_features):
    units = 128
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))

    enc_inp = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='causal', input_shape=(n_past, n_features))(
        encoder_inputs)  # BiB: added conv layer

    encoder_outputs1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, activation='relu', return_state=True))(
        enc_inp)
    x = tf.keras.layers.Dense(2 * units * n_future)(encoder_outputs1[0])  # BiB: instead of repeatvector
    decoder_inputs = tf.keras.layers.Reshape([n_future, 2 * units])(x)

    # decoder_inputs = tf.keras.layers.RepeatVector(n_future)(tf.keras.layers.Dense(256)(encoder_outputs1[0])

    decoder_l1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, activation='relu', return_sequences=True))(
        decoder_inputs, initial_state=encoder_outputs1[1:])
    # decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation='relu'))(
    # decoder_l1) #BiB: suppressed TD layer
    decoder_outputs1 = tf.keras.layers.Dense(n_features, activation='linear')(
        decoder_l1)  # BiB: replaced above with this

    decoder_outputs1 = decoder_outputs1 + tf.keras.layers.RepeatVector(n_future)(
        tf.reduce_mean(encoder_inputs, axis=1))  # BiB: added to make residual net

    model_e1d1 = tf.keras.models.Model(encoder_inputs, decoder_outputs1)

    model_e1d1.summary(line_length=350)
    return model_e1d1


callback_es_train = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
callback_es_val = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
lr = 0.00001  # 0.00001
optimiser = tf.keras.optimizers.Adam(learning_rate=lr)
# # # # *************************LearningRate*******************************************************
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
# # # # *************************LearningRate*******************************************************

# BiB: save/reload best weights
WEIGHTS_PATH = 'best_weights_all.hdf5'
# WEIGHTS_PATH = './best_weights.hdf5'
NUM_REPEATS = 5  # BiB: multiple restarts may be needed to find good fit
if exists(WEIGHTS_PATH):
    model = model1_bi(n_past=n_past, n_features=n_features)
    model.load_weights(WEIGHTS_PATH)
else:
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='best_weights_all.hdf5', monitor='loss',
                                                      verbose=1,
                                                      save_best_only=True, save_weights_only=False, mode='auto',
                                                      save_freq=train_size)
    for k in range(NUM_REPEATS):
        model = model1_bi(n_past=n_past, n_features=n_features)  # BiB: reset model
        model.compile(optimizer=optimiser, loss='mse', metrics=['mae'])
        history_model = model.fit(train, epochs=100, validation_data=test, verbose=1, shuffle=True,
                                  callbacks=[callback_es_train, callback_es_val, checkpointer])
        print(f"Done pass {k}")

    fig1, ax1 = plt.subplots()
    ax1.plot(history_model.history['loss'], 'r', label='TrainLoss')
    ax1.plot(history_model.history['val_loss'], 'y', label='ValidationLoss')
    ax1.legend()

# print(type(df_pred), df_pred)

# fig2, ax2 = plt.subplots()
# ax2.plot(sub_df.n_lng, sub_df.e_lat, '.')
# ax2.plot(df_pred.lng, df_pred.lat, 'r*', label=f'my predicted:{n_future}')
# ax2.plot(x_test.n_lng, x_test.e_lat, 'yo', label=f'past data:{n_past}')
# ax2.legend()

# BiB: Added a 3rd plot for expected output to compare w/ prediction
# fig3, ax3 = plt.subplots()
# y_test = sub_df[start_p + n_past: start_p + n_past + n_future]
# ax3.plot(y_test.n_lng, y_test.e_lat, 'r*', label=f'true future:{n_future}')
# ax3.plot(x_test.n_lng, x_test.e_lat, 'yo', label=f'past data:{n_past}')
# ax3.legend()

# print(df_pred, y_test)

# plt.show()

# # ********************************Animation *****************************************

# file_not_trained = 'Data/logs/20220429-171330--lm_1p_fd4-IVER3-3072.log'


# file_not_trained = 'Data/logs/20220429-151241--lm_1p-IVER3-3072.log'
file_not_trained = 'Data/logs/20220429-161912--lm_1p_fd2-IVER3-3072.log'
# file_not_trained = 'Data/logs/20220429-171330--lm_1p_fd4-IVER3-3072.log'
df_ani = pd.read_csv(file_not_trained, delimiter=';', skiprows=0)

scaler_ani = MinMaxScaler(feature_range=(-1., 1.))
df_ani = preprocessing(df_ani, scaler_ani)
# print(df_ani.head(5))
fig, ax = plt.subplots()


def init():
    ax.plot(df_ani.n_lng, df_ani.e_lat, alpha=0.5)
    ax.set_xlabel('X axis', fontsize=12)
    ax.set_ylabel('Y axis', fontsize=12)
    ax.legend()
    return ax,


start_p = 10


def animate(i):
    # time.sleep(0.1)
    i = i + start_p
    scaled = df_ani[i: i + n_past]
    data_ani = np.array(scaled)
    data_ani = data_ani.reshape((1, -1, n_features))
    pred = model.predict(data_ani)
    pred = np.squeeze(pred, axis=0)
    df_pred = pd.DataFrame(pred, columns=(['lat', 'lng', 'dt']))
    pred = scaler_ani.inverse_transform(df_pred)
    df_pred.drop('dt', axis=1, inplace=True)
    print(i)
    plt.cla()
    ax.plot(df_ani.n_lng, df_ani.e_lat, alpha=0.5)
    ax.plot(scaled.n_lng, scaled.e_lat, 'y', label='past')
    ax.plot(df_pred.lng, df_pred.lat, 'r', label='predicted')
    return ax


# ani = FuncAnimation(plt.gcf(), animate, interval=10)
ani = FuncAnimation(plt.gcf(), animate, init_func=init, interval=10)
plt.tight_layout()
plt.show()
