import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def trend(time, slope=0):
    return slope * time


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def seasonal_pattern(season_time):
    return np.where(season_time < 0.1,
                    np.cos(season_time * 7 * np.pi),
                    1 / np.exp(5 * season_time))


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def plot_series(time, series, format="-", title="", label=None, start=0, end=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    if label:
        plt.legend()
    plt.grid(True)


TIME = np.arange(4 * 365 + 1, dtype="float32")
y_intercept = 10
slope = 0.01
SERIES = trend(TIME, slope) + y_intercept
# Adding seasonality
amplitude = 40
SERIES += seasonality(TIME, period=365, amplitude=amplitude)
# Adding noise
noise_level = 2
SERIES += noise(TIME, noise_level, seed=42)
plt.figure(figsize=(10, 6))
plot_series(TIME, SERIES)
plt.show()


SPLIT_TIME = 1100
def split_time(time, series, time_step=SPLIT_TIME):
    train_time = time[:SPLIT_TIME]
    train_series = series[:SPLIT_TIME]
    val_time = time[SPLIT_TIME:]
    val_series = series[SPLIT_TIME:]
    return train_time, train_series, val_time, val_series


train_time, train_series, val_time, val_series = split_time(TIME, SERIES)
plt.figure(figsize=(10, 6))
plot_series(train_time, train_series, title="Training")
plt.show()
# -----------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plot_series(val_time, val_series, title="Validation")
plt.show()


def compute_metrics(true_series, forecast):
    mse = tf.keras.metrics.mean_squared_error(true_series, forecast)
    mae = tf.keras.metrics.mean_absolute_error(true_series, forecast)
    return mse, mae


zeros = np.zeros(5)
ones = np.ones(5)
mse, mae = compute_metrics(zeros, ones)
print(f'mse 1:{mse} dan mae 1:{mae}')
mse, mae = compute_metrics(ones, ones)
print(f'mse 2: {mse} dan mae 2: {mae}')
print(f'Tipe data mse: {np.issubdtype(type(mse), np.number)}')
# NAIVE FORECAST
naive_forecast = SERIES[SPLIT_TIME -1:-1]
plt.figure(figsize=(10, 6))
plot_series(val_time, val_series, label='validation set')
plot_series(val_time, naive_forecast, label='naive forecast')
plt.show()
plot_series(val_time, val_series, start=50, end=70, label='validation set')
plot_series(val_time, naive_forecast, start=50, end=70, label='naive forecast')
plt.show()
mse, mae = compute_metrics(val_series, naive_forecast)
print(f'mse : {mse} dan mae : {mae}')


def moving_average_forecast(series, window_size):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    numpy_forecast = np.array(forecast)
    return numpy_forecast


move = moving_average_forecast(SERIES, window_size=30)
print(f'shape{move.shape}')
move = move[1100 - 30:]
print(f'kompatibel dengan validation series {val_series.shape == move.shape}')
plt.figure(figsize=(10, 6))
plot_series(val_time, val_series)
plot_series(val_time, move)
plt.show()
mse, mae = compute_metrics(val_series, move)
print(f'mse : {mse} dan mae: {mae}')
diff_series = (SERIES[365:] - SERIES[:-365])
diff_time = TIME[365:]
print(f'SERIES memiliki {len(SERIES)} data dan perbedaannya harus memiliki {len(SERIES) - 365} data')
plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show()
diff_move = moving_average_forecast(diff_series, 50)
diff_move = diff_move[SPLIT_TIME - 365 -50:]
print(f'kompatibel {val_series.shape == diff_move.shape}')
plt.figure(figsize=(10, 6))
plot_series(val_time, diff_series[1100 - 365:])
plot_series(val_time, diff_move)
plt.show()
past_series = (SERIES[SPLIT_TIME - 365:-365])
diff_move_plus_past = past_series + diff_move
plot_series(val_time, val_series)
plot_series(val_time, diff_move_plus_past)
plt.show()
mse, mae = compute_metrics(val_series, diff_move_plus_past)
print(f'mse : {mse} dan mae : {mae}')