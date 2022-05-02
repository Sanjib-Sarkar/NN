import matplotlib.colors as pcolor
import numpy as np
import pandas
import pandas as pd
import pymap3d as pm
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from os.path import exists

tf.keras.backend.clear_session()

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
indices = [*range(end - start)]  # BiB: added an index feature to data
sub_df['ind'] = indices


def scaling(data: pandas.DataFrame):  # BiB: changed data -> data1, probably unnecessary
    mn_scaler = MinMaxScaler(feature_range=(-1., 1.))  # BiB: changed range
    data1 = data.copy()
    mn_scaler.fit(data1)
    data1 = mn_scaler.transform(data1)
    data1 = pd.DataFrame(data1, columns=(['lat', 'lng', 'ind']))
    return mn_scaler, data1


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
n_future = 30
n_features = 2 + 1  # BiB: 3rd feature added
batch_size = 32
mn_scaler, scaled_data = scaling(sub_df)
# print(type(scaled_data))

shuffled_data = sequence_data(scaled_data, window_size=n_past, forecast_size=n_future, batch_size=batch_size)

total_size = len(list(shuffled_data.as_numpy_iterator()))

percentage = 0.80
train_size = int(total_size * percentage)
# test_size = 1 - train_size #BiB: suppressed size that was using all data for test anyway

train = shuffled_data.take(train_size)
test = shuffled_data.shuffle(total_size).take(-1)  # BiB: using all data for test, as before

steps = 0
steps_on_past = 0


# # n_past = int(n_past / steps)
# n_future = int(n_future / steps)  # 5 steps

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
    # decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation='relu'))(decoder_l1) #BiB: suppressed TD layer
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
WEIGHTS_PATH = './best_weights.hdf5'
NUM_REPEATS = 5  # BiB: multiple restarts may be needed to find good fit
if exists(WEIGHTS_PATH):
    model = model1_bi(n_past=n_past, n_features=n_features)
    model.load_weights(WEIGHTS_PATH)
else:
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='./best_weights.hdf5', monitor='loss', verbose=1,
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

# plt.show()

# start_p = 5500
start_p = 1300  # 1450
x_test = sub_df[start_p: start_p + n_past]
# x_scaler, x_scaled_test = scaling(x_test) #BiB: suppressed local scaling
x_scaled_test = scaled_data[start_p: start_p + n_past]  # BiB: use globally scaled data
x_scaled_test = np.array(x_scaled_test)
x_scaled_test = x_scaled_test.reshape((1, -1, n_features))

# print(type(x_test), x_test[:5])
pred = model.predict(x_scaled_test)
pred = np.squeeze(pred, axis=0)

df_pred = pd.DataFrame(pred, columns=(['lat', 'lng', 'ind']))  # BiB: added 3rd feature

# df_pred = x_scaler.inverse_transform(df_pred)
df_pred = mn_scaler.inverse_transform(df_pred)  # BiB: used global scaler

df_pred = pd.DataFrame(df_pred, columns=(['lat', 'lng', 'ind']))
df_pred.drop('ind', axis=1, inplace=True)  # BiB: dropped extra feature
# print(type(df_pred), df_pred)

fig2, ax2 = plt.subplots()
ax2.plot(sub_df.n_lng, sub_df.e_lat, '.')
ax2.plot(df_pred.lng, df_pred.lat, 'r*', label=f'my predicted:{n_future}, sample:{steps}')
ax2.plot(x_test.n_lng, x_test.e_lat, 'yo', label=f'past data:{n_past}, sample:{steps_on_past}')
ax2.legend()

# BiB: Added a 3rd plot for expected output to compare w/ prediction
fig3, ax3 = plt.subplots()
y_test = sub_df[start_p + n_past: start_p + n_past + n_future]
ax3.plot(y_test.n_lng, y_test.e_lat, 'r*', label=f'true future:{n_future}, sample:{steps}')
ax3.plot(x_test.n_lng, x_test.e_lat, 'yo', label=f'past data:{n_past}, sample:{steps_on_past}')
ax3.legend()

print(df_pred, y_test)
plt.show()
