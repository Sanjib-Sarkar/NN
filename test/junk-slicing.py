import numpy as np
from keras_preprocessing.sequence import TimeseriesGenerator
from numpy import random

random.seed(42)
numbers = np.random.randint(0, 20, (10, 2))


def ts_data_preparation(features, window=5, future=1, batch=1, shuffle=False):
 target = features[(future - 1):]
 features = features[:-(future - 1)] if future != 1 else features
 tensor = TimeseriesGenerator(features, target, length=window_size, batch_size=batch_size, shuffle=shuffle)
 return tensor


window_size = 4
future = 1
batch_size = 1

print(numbers)
print(ts_data_preparation(numbers)[0])


# coordinates = [[ 24.89923663 -10.82196907]
#  [ 25.76644998 -10.49604334]
#  [ 26.94997571 -10.33640438]
#  [ 28.14023227 -10.49936322]
#  [ 28.90841895 -10.63349988]
#  [ 30.59189233 -11.06030014]
#  [ 31.3802701  -11.50262294]
#  [ 31.96193916 -11.8462823 ]
#  [ 32.66763362 -12.28084515]
#  [ 33.70982997 -12.59346268]
#  [ 35.03564897 -12.6832538 ]
#  [ 36.36339044 -12.64888335]
#  [ 37.856498   -12.4105327 ]
#  [ 39.29864937 -12.15111893]
#  [ 41.90125003 -11.24983041]
#  [ 43.21456665 -10.42282159]
#  [ 44.31252064  -9.33307907]
#  [ 45.06435796  -8.23668636]
#  [ 45.56430029  -7.37198873]
#  [ 46.10942928  -6.34876336]]