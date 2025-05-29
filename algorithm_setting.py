import numpy as np
from filterpy.kalman import KalmanFilter

# Low-pass filter
def lowpass_filter(data, alpha=0.2):
    data = np.asarray(data)
    filtered = np.zeros_like(data)
    filtered[0] = data[0]
    for i in range(1, len(data)):
        filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i - 1]
    return filtered

# Initialize Kalman filter
def init_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 1000.
    kf.R = np.array([[0.05, 0],
                     [0, 0.05]])
    kf.Q = np.eye(4) * 0.01
    return kf

# Moving average smoothing
def moving_average(history, new_point, window_size=5):
    history.append(new_point)
    if len(history) > window_size:
        history.pop(0)
    return np.mean(history)

# Performance evaluation class
class PerformanceEvaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = {
            'lowpass': {'positions': [], 'errors': [], 'jitter': []},
            'kalman': {'positions': [], 'errors': [], 'jitter': []},
            'moving_avg': {'positions': [], 'errors': [], 'jitter': []}
        }

    def record(self, algo_name, position, error):
        positions = self.data[algo_name]['positions']
        if positions:
            prev_pos = positions[-1]
            jitter = np.sqrt((position[0] - prev_pos[0])**2 + (position[1] - prev_pos[1])**2)
            self.data[algo_name]['jitter'].append(jitter)
        self.data[algo_name]['positions'].append(position)
        self.data[algo_name]['errors'].append(error)

    def get_metrics(self):
        metrics = {}
        for algo in self.data:
            errors = self.data[algo]['errors']
            jitter = self.data[algo]['jitter']
            metrics[algo] = {
                'avg_error': np.mean(errors) if errors else 0,
                'max_error': np.max(errors) if errors else 0,
                'avg_jitter': np.mean(jitter) if jitter else 0,
                'max_jitter': np.max(jitter) if jitter else 0
            }
        return metrics