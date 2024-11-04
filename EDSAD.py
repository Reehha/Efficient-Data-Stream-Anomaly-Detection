import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

def data_stream_generator(trend_type='linear', seasonality_period=50, seasonality_amplitude=10, noise_level=1, delay=0.1):
    """A generator function to yield continuous data points with irregular patterns."""
    index = 0
    trend_shifted = False  # Flag to indicate if the trend has been shifted
    while True:
        # Generate normal trend and seasonal components
        if trend_type == 'linear':
            trend = 0.01 * index
        elif trend_type == 'sinusoidal':
            trend = 5 * np.sin(0.01 * index)
        else:
            raise ValueError("Invalid trend_type. Choose 'linear' or 'sinusoidal'.")
        
        seasonality = seasonality_amplitude * np.sin(2 * np.pi * index / seasonality_period)
        noise = np.random.normal(0, noise_level)

        # Introduce irregularities in the pattern
        if index % 50 == 0 and not trend_shifted:  # Every 50 points, shift the trend
            trend_shifted = True
            trend += np.random.normal(30, 5)  # Abrupt change in trend

        if index % 75 == 0:  # Every 75 points, change seasonality amplitude
            seasonality_amplitude = np.random.uniform(5, 20)

        if np.random.rand() < 0.05:  # 5% chance of a random spike or drop
            noise += np.random.choice([-50, 50])  # Sudden spike or drop

        data_point = trend + seasonality + noise
        yield data_point

        index += 1
        time.sleep(delay)

def is_pattern_unusual(window, threshold=1):
    """Determine if the mean of the window deviates from the overall mean."""
    mean_window = np.mean(window)
    std_dev_window = np.std(window)
    
    # Check if the window mean is outside the threshold
    return abs(mean_window) > threshold * std_dev_window  # Check if the mean is an anomaly based on its own std dev

def moving_average(data, window_size):
    """Calculate the moving average for the data."""
    if len(data) < window_size:
        return None
    return np.mean(list(data)[-window_size:])  # Convert deque to list for slicing

def visualize_stream(data_generator, max_points=200, prediction_window=5, pattern_window_size=50):
    """Visualizes data from a generator in real-time, flags anomalies, and detects unusual patterns."""
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    data_queue = deque(maxlen=max_points)
    anomalies = []
    unusual_patterns = []
    line, = ax.plot([], [], 'b-', label='Data Stream')  # Normal data
    anomaly_marker, = ax.plot([], [], 'ro', markersize=5, label='Anomalies')  # Red circles for anomalies
    pattern_marker, = ax.plot([], [], 'go', markersize=5, label='Unusual Patterns')  # Green circles for unusual patterns

    while True:
        try:
            data_point = next(data_generator)
            data_queue.append(data_point)

            # Initialize mean and std_dev
            mean = None
            std_dev = None

            # Check for anomalies only if we have enough data points
            if len(data_queue) > 5:  # Ensure we have enough data points
                mean = np.mean(data_queue)
                std_dev = np.std(data_queue)

                if abs(data_point - mean) > 3 * std_dev:  # Simple anomaly detection
                    anomalies.append(data_point)
                    print(f"Anomaly detected: {data_point:.2f}")

            # Detect unusual patterns using a sliding window
            if len(data_queue) >= pattern_window_size:
                if is_pattern_unusual(list(data_queue)[-pattern_window_size:]):
                    unusual_patterns.append(data_point)
                    print(f"Unusual pattern detected: {data_point:.2f}")

            # Update the plot with new data
            line.set_xdata(np.arange(len(data_queue)))
            line.set_ydata(data_queue)

            # Only update anomaly markers if mean and std_dev are valid
            if mean is not None and std_dev is not None:
                anomaly_x = np.arange(len(data_queue))
                anomaly_y = [point if abs(point - mean) > 3 * std_dev else np.nan for point in data_queue]
                anomaly_marker.set_xdata(anomaly_x)
                anomaly_marker.set_ydata(anomaly_y)

            # Update unusual pattern markers
            pattern_x = np.arange(len(data_queue))
            pattern_y = [point if point in unusual_patterns else np.nan for point in data_queue]
            pattern_marker.set_xdata(pattern_x)
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

# Call the visualization function
visualize_stream(stream, max_points=200, prediction_window=5, pattern_window_size=50)

# Documentation of Algorithm Choice
"""
Anomaly Detection Algorithm:
The implemented algorithm uses a simple Z-score method to detect anomalies, which flags data points that are more than 3 standard deviations away from the mean. 
This approach is effective for identifying significant deviations in data, particularly when the underlying distribution is Gaussian.
However, it may require adjustments or a more sophisticated model (e.g., Isolation Forest) for data that exhibit concept drift or more complex seasonal patterns.
"""
