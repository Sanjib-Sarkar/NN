import matplotlib.colors as pcolor
import numpy as np
import pandas as pd
import pymap3d as pm
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

plt.style.use('dark_background')
colors = [k for k, v in pcolor.cnames.items()]

file = 'Data/20220222_153859_hlc.csv'
df = pd.read_csv(file, skiprows=1)
df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng'}, errors="raise")

df = df.loc[(df['lat'] != 0)]
df['h'] = 0
df.reset_index(drop=True, inplace=True)
# print(df.head(5))
index = int(len(df.lat) / 2)
t_start = 12700
taj_lat = df.lat[t_start: t_start+125]
taj_lng = df.lng[t_start: t_start+125]
taj = np.array([[i, j] for i, j in zip(taj_lat, taj_lng)])
print('Taj:', taj.shape)
' Convert to local coordinate system'
df['e_lat'], df['n_lng'], df['u'] = pm.geodetic2enu(df.lat, df.lng, df.h, df.lat[index],
                                                    df.lng[index], df.h, ell=None, deg=True)

start = 10450
end = 13820
# df = df.iloc[::2]
lat = df.lat[start:end]
lng = df.lng[start:end]

series_time = df.Time[start:end].to_list
series_lat = df.e_lat[start:end]
series_lng = df.n_lng[start:end]

coordinates = np.array([[i, j] for i, j in zip(series_lat, series_lng)])


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


scaler, coordinates = scaling(coordinates)

train, test = train_test_split(coordinates, 0.8)

window_size = 90
future = 2
batch_size = 32

features = train[:-(window_size + future)]
target = train[(window_size + future):]

print(f'Train: {train.shape}, Features: {features.shape}, Targets: {target.shape}')

xy_train = ts_from_array(features, target, window=window_size, batch=batch_size, shuffle=True)


def model_lstm(w_size):
    units = 8
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(w_size * 2, return_sequences=True, input_shape=(w_size, train.shape[1])),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(2, activation='linear')])
    model.summary()
    return model


model = model_lstm(window_size)
lr = 0.001
# optimiser = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
optimiser = tf.keras.optimizers.Adam(learning_rate=lr)

# lr_schedular = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr * 10 ** (epoch / 25))
# es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
# model.compile(optimizer=optimiser, loss='mse', metrics=['mae'])
# epochs = 100
# history = model.fit(xy_train, epochs=epochs, verbose=1, callbacks=[lr_schedular, es_callback])
# plt.semilogx(history.history["lr"], history.history["loss"], label='loss')
# # plt.axis([1e-3, 1, 0, 0.4])
# plt.xlabel('Learning rate')
# plt.legend()
# plt.show()

model.compile(optimizer=optimiser, loss='mse', metrics=['accuracy'])
epochs = 30

valid_target = test[(window_size + future):]
valid_features = test[:-(window_size + future)]
xy_validation = ts_from_array(valid_features, valid_target, window=window_size, batch=batch_size, shuffle=False)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
history = model.fit(xy_train, epochs=epochs, validation_data=xy_validation, verbose=1, callbacks=callback)

# print(history.history.keys())
# # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
fig1, ax1 = plt.subplots()
ax1.plot(history.history['loss'], 'r', label='TrainLoss')
# ax1.plot(history.history['accuracy'], 'orange', label='TrainAccuracy')
ax1.plot(history.history['val_loss'], 'w', label='ValidationLoss', zorder=1)
# ax1.plot(history.history['val_accuracy'], 'g', label='ValidationAccuracy')
ax1.set_xlabel('Epochs')
# plt.grid(visible=True, axis='both')
ax1.set_title(f'Window:{window_size}, future: {future}')
ax1.legend()
# plt.show()

test = ts_from_array(coordinates, None, window=window_size, batch=1, shuffle=False)

predicted = model.predict(test)
# print(f'Predicted:{predicted}: type{type(predicted)}')
df_n = pd.DataFrame(predicted, columns=('e', 'n'))
predicted_coor = scaler.inverse_transform(df_n)

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
# plt.show()
# ************************************************************************************************
origin_lat = df.lat[index]
origin_lng = df.lng[index]
# print(f'OriginLat: {origin_lat}, OriginLng: {origin_lng}')
# t_prediction = test[:window_size]
# print(f't_prediction: {t_prediction.shape}: {type(t_prediction)}, {t_prediction[:3]}')
# mn, data = scaling(t_prediction)
# print(f't_prediction: {data.shape}: {type(data)}, {data[:3]}')
# data = mn.inverse_transform(data)
# print(f't_prediction: {data.shape}: {type(data)}, {data[:3]}')
# print(data[:, :1], data[:, 1:])
# back_lat, back_lng, b_u = pm.geodetic2enu(data[:, :1], data[:, 1:], 0, origin_lat, origin_lng, 0, ell=None, deg=True)
# print(f'back: {back_lat[:3]},{back_lng[:3], b_u[:3]}')
# taj = taj[-20:]
# print("taj:", taj, taj.shape)
next = 30
path = np.zeros(shape=(next, 2))
for i in range(next):
    ' Convert to local coordinate system'
    origin_lat = taj[-1, 0:1]
    origin_lng = taj[-1, -1:]
    # print(f'taj:{taj[:1]},lat:{origin_lat}, lng:{origin_lng}')
    t_lat, t_lng, _ = pm.geodetic2enu(taj[-window_size:, :1], taj[-window_size:, 1:], 0, origin_lat, origin_lng, 0, ell=None, deg=True)
    l_co = np.hstack((t_lat, t_lng))
    scaler, coordinates = scaling(l_co)
    # xy_pre = ts_from_array(coordinates, None, window=window_size, batch=1, shuffle=False)
    xy_pre = np.reshape(coordinates, (1, coordinates.shape[0], coordinates.shape[1]))
    predicted = model.predict(xy_pre)  # Predicted:[[-0.19003247  0.02555241]]: type<class 'numpy.ndarray'>
    p_cor = scaler.inverse_transform(predicted)
    # print(f'{l_co}')
    lat, lng, _ = pm.enu2geodetic(p_cor[:, :1], p_cor[:, 1:], 0, origin_lat, origin_lng, 0, ell=None, deg=True)
    co = np.hstack((lat, lng))
    # print('Co:', co)
    path[[i]] = co
    i += 1
    taj = np.append(taj, co, axis=0)


# print(f'path:{path}')

plt.scatter(taj[:, -1:], taj[:, 0:1], marker='*', color='c')
# plt.scatter(path[:, -1:], path[:, 0:1], s=80, facecolors='none', edgecolors='y')
plt.plot(path[:, -1:], path[:, 0:1], color='y')
plt.show()
# class Scaling:
#     def __init__(self):
#         self.mn_scaler = MinMaxScaler(feature_range=(-1, 1))
#         self.sd_scaler = StandardScaler()
#
#     def mn_sc(self, data):
#         self.mn_scaler.fit(data)
#         data = self.mn_scaler.transform(data)
#         return self.mn_scaler, data
#
#     def sd_sc(self, data):
#         self.sd_scaler.fit(data)
#         data = self.sd_scaler.transform(data)
#         return self.sd_scaler, data
#
#     def mn_sc_back(self, data):
#         data = self.mn_scaler.inverse_transform(data)
#         return data
#
#     def sd_sc_back(self, data):
#         data = self.sd_scaler.inverse_transform(data)
#         return data
#
#
# class TsFromArray:
#     def __init__(self):
#         self.sequence_stride = 1
#         self.seed = None
#         self.start_index = None
#         self.end_index = None
#
#     def ts_from_array(self, features, targets, window, batch, shuffle):
#         print("Features and targets shape: ", features.shape, targets.shape if targets is not None else '')
#         tensor = tf.keras.preprocessing.timeseries_dataset_from_array(features, targets, sequence_length=window,
#                                                                       sequence_stride=self.sequence_stride,
#                                                                       sampling_rate=1, batch_size=batch,
#                                                                       shuffle=shuffle,
#                                                                       seed=self.seed, start_index=self.start_index,
#                                                                       end_index=self.end_index)
#         return tensor
#
#
# class RecurrentModels:
#     def __init__(self):
#         self.model = None
#         # self.w_size = 16
#         self.unit = 8
#         self.initial_lr = 1e-6
#         self.optimiser = tf.keras.optimizers.Adam(learning_rate=self.initial_lr)
#         # tf.keras.backend.clear_session()
#
#     def model_lstm(self, w_size, train_shape_1=2):
#         self.model = tf.keras.models.Sequential([
#             tf.keras.layers.LSTM(w_size * 2, return_sequences=True, input_shape=(w_size, train_shape_1)),
#             tf.keras.layers.LSTM(self.unit),
#             tf.keras.layers.Dense(2, activation='linear')])
#         # model.summary()
#         return self.model
#
#     def model_lstm2(self):
#         pass
#
#     def tune_learning_rate(self, model, xy_train_data):
#         es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
#         lr_schedular = tf.keras.callbacks.LearningRateScheduler(lambda epoch: self.initial_lr * 10 ** (epoch / 25))
#         model.compile(optimizer=self.optimiser, loss='mse', metrics=['mae'])
#         history_lr = model.fit(xy_train_data, epochs=100, verbose=1, callbacks=[lr_schedular, es_callback])
#         return history_lr
#
#     def model_fit(self, model, learning_rate, epchs):
#         optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#         model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
#         c_back = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=8)
#         hs = model.fit(xy_train, epochs=epchs, validation_data=xy_validation, verbose=1, callbacks=c_back)
#         return hs
#
#     def model_predict(self, model, data, origin):
#         predicted = model.predict(test)
#         return predicted
#
#
# def read_file(filename):
#     dframe = pd.read_csv(filename, skiprows=1)
#     dframe = dframe.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng'}, errors="raise")
#     dframe = dframe.loc[(df['lat'] != 0)]
#     dframe['h'] = 0
#     dframe.reset_index(drop=True, inplace=True)
#     # print(df.head(5))
#     index = int(len(dframe.lat) / 2)
#     return dframe, index


# if __name__ == '__main__':
#
#
#     file = 'Data/20220222_153859_hlc.csv'
#     df, index = read_file(file)
#
#     scaler = Scaling
#     df =
#
