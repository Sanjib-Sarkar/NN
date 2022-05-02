import tensorflow as tf
import pandas as pd


def create_dataset(df: pd.DataFrame, features,
                   window_size, forecast_size,
                   batch_size):
    # Feel free to play with shuffle buffer size
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
    data = data.map(lambda k: (k[:-forecast_size:2], k[-forecast_size::2]))

    return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


coordinates = {'lat': [i for i in range(100)], 'lng': [i for i in range(100)]}

df = pd.DataFrame(coordinates)

df_sub = create_dataset(df, 0, window_size=5, forecast_size=5, batch_size=1)


print(f'ttype: {type(df_sub)}, {list(df_sub.as_numpy_iterator())[0]}')
total_size = len(list(df_sub.as_numpy_iterator()))

percentage = 0.8
test_size = int(total_size * percentage)

print('total size', total_size, test_size)
train = df_sub.take(60)
test = df_sub.take(20)
train_data = list(train.as_numpy_iterator())
print('list: ', train_data[0], len(train_data))
print(f'test:{list(test.as_numpy_iterator())[0]}')
