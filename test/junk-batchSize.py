import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd

file = 'Data/20220222_153859_hlc.csv'
df = pd.read_csv(file, skiprows=1)
df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng'}, errors="raise")
df = df.loc[(df['lat'] != 0)]


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
    return tensor


window_size = 3
future = 2
batch_size = 5

start = 10500
end = 13700
series_lat = df.lat[start:end].to_numpy()

train_ts_lat = ts_data_preparation(series_lat, window_size=window_size, future=future, batch_size=batch_size,
                                   shuffle=False)

wd = windowed_dataset(series_lat, window_size=window_size, batch_size=batch_size,
                      shuffle_buffer=1)

print("series_lat: ", series_lat[:10], len(series_lat))
print(" train_x_lat: ", train_ts_lat[0])
# print('Size of wd', len(wd))
i = 0
for example in wd:
    print('X' * 100)
    print(example[0].numpy())
    print(example[1].numpy())
    i += 1

print(i)
