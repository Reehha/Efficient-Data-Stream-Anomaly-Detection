import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque

def data_stream_generator(trend_type='linear', seasonality_period=50, seasonality_amplitude=10, noise_level=1, delay=0.1):
    index = 0
    trend_shifted = False
    while True:
        if trend_type == 'linear':
            trend = 0.01 * index
        elif trend_type == 'sinusoidal':
            trend = 5 * np.sin(0.01 * index)
        else:
            raise ValueError("Invalid trend_type. Choose 'linear' or 'sinusoidal'.")
        
        seasonality = seasonality_amplitude * np.sin(2 * np.pi * index / seasonality_period)
        noise = np.random.normal(0, noise_level)

        if index % 50 == 0 and not trend_shifted:
            trend_shifted = True
            trend += np.random.normal(30, 5)

        if index % 75 == 0:
            seasonality_amplitude = np.random.uniform(5, 20)

        if np.random.rand() < 0.05:
            noise += np.random.choice([-50, 50])

        yield trend + seasonality + noise
        index += 1
        time.sleep(delay)

def visualize_stream(stream, max_points=200, prediction_window=5, pattern_window_size=50):
    data_queue = deque(maxlen=max_points)
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='Data Stream')
    anomaly_marker, = ax.plot([], [], 'ro', label='Anomalies')
    pattern_marker, = ax.plot([], [], 'go', label='Unusual Patterns')

    while True:
        try:
            data_point = next(stream)
            data_queue.append(data_point)

            data_x = np.arange(len(data_queue))
            data_y = np.array(data_queue)
            line.set_xdata(data_x)
            line.set_ydata(data_y)

            mean = np.mean(data_y)
            std_dev = np.std(data_y)
            anomaly_y = [point if abs(point - mean) > 3 * std_dev else np.nan for point in data_queue]
            anomaly_marker.set_xdata(data_x)
            anomaly_marker.set_ydata(anomaly_y)

            unusual_patterns = set(data_y[-pattern_window_size:])
            pattern_y = [point if point in unusual_patterns else np.nan for point in data_queue]
            pattern_marker.set_xdata(data_x)
            pattern_marker.set_ydata(pattern_y)

            ax.relim()
            ax.autoscale_view()
            ax.legend()
            plt.draw()
            plt.pause(0.01)

        except StopIteration:
            print("Data stream has ended.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

# Example usage
stream = data_stream_generator(trend_type='sinusoidal', seasonality_period=100, seasonality_amplitude=15, noise_level=3, delay=0.1)
visualize_stream(stream, max_points=200, prediction_window=5, pattern_window_size=50)
