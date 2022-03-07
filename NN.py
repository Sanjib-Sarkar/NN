import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

file = 'Data/20220222_153859_hlc.csv'
df = pd.read_csv(file, skiprows=1)
df = df.rename(columns={"Latitude (Deg N)": 'lat', "Longitude (Deg W)": 'lng'}, errors="raise")
# print(df.columns)
# print(df.head(5))
df = df.loc[(df['lat'] != 0)]
print(len(df.lat))
start = 12400
end = 13700
series_time = df.Time[start:end]
series_lat = df.lat[start:end]
series_lng = df.lng[start:end]
# print(type(series_lng))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(df['lng'], df['lat'])
ax2.plot(series_lng, series_lat)

fig2, (axa, axb, axc) = plt.subplots(3, 1, figsize=(10, 6))
axa.plot(series_time, series_lng)
axa.set_xticks(axa.get_xticks()[::240])
axa.set_title('Longitude')
axb.plot(series_time, series_lat)
axb.set_xticks(axb.get_xticks()[::240])
axb.set_title('Latitude')

color = 'red'
axc.plot(series_time, series_lng, color=color)
axc.set_ylabel('Longitude', color=color)
axc.set_xticks(axc.get_xticks()[::240])

axy = axc.twinx()  # instantiate a second axes that shares the same x-axis
axy.plot(series_time, series_lat)
axy.set_ylabel('Latitude')


split_time = 500
lng_train = series_lng[:split_time]
lng_valid = series_lng[split_time:]

lat_train = series_lat[:split_time]
lat_valid = series_lat[split_time:]

naive_forecast_lng = series_lng[split_time - 1:-1]
naive_forecast_lat = series_lat[split_time - 1:-1]

fig3, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(lng_valid)
ax[0].plot(naive_forecast_lng)
print(keras.metrics.mean_squared_error(lng_valid, naive_forecast_lng).numpy())
print(keras.metrics.mean_absolute_error(lng_valid, naive_forecast_lng).numpy())
plt.show()

# naive_forecast = series[split_time - 1:-1]
