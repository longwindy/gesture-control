import numpy as np
from filterpy.kalman import ExtendedKalmanFilter, KalmanFilter

class LowPassFilter:
    def __init__(self, alpha=0.2):
        """
        Initialize the low-pass filter.

        :param alpha: Filter coefficient, range [0, 1], default value is 0.2.
        """
        self.alpha = alpha
        self.prev_value = None

    def filter(self, data):
        """
        Apply low-pass filtering to the input data.

        :param data: Input data.
        :return: Filtered data.
        """
        if self.prev_value is None:
            self.prev_value = data
        filtered = self.alpha * data + (1 - self.alpha) * self.prev_value
        self.prev_value = filtered
        return filtered

class MovingAverageFilter:
    def __init__(self, window_size=5):
        """
        Initialize the moving average filter.

        :param window_size: Size of the sliding window, default value is 5.
        """
        self.window_size = window_size
        self.history = []

    def filter(self, new_point):
        """
        Apply moving average filtering to the new data point.

        :param new_point: New data point.
        :return: Filtered value.
        """
        self.history.append(new_point)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return np.mean(self.history)

class ExtendedKalmanFilterWrapper:
    def __init__(self, dt=1/30.0):
        self.ekf = ExtendedKalmanFilter(dim_x=5, dim_z=2)  # State dimension changed to 5 [x, y, v, θ, ω]
        # Initial state [x, y, velocity, heading, turn_rate]
        self.ekf.x = np.array([0., 0., 0., 0., 0.])
        # Adjust the initial state covariance matrix to reflect the uncertainty of different states
        self.ekf.P = np.diag([100, 100, 10, 100, 100])  # Greater uncertainty in position and angle
        # Observation noise (keep 2x2)
        self.ekf.R = np.array([[0.05, 0], [0, 0.05]])
        # Optimize the process noise matrix
        self.ekf.Q = np.diag([0.1, 0.1, 0.5, 1.0, 1.0])  # Greater noise in velocity, angle, and angular velocity
        self.ekf.dt = dt

    def f(self, x):
        dt = self.ekf.dt
        v, θ, ω = x[2], x[3], x[4]
        
        # CTRV motion model
        if abs(ω) < 1e-5:  # Handle the case of zero angular velocity
            return np.array([
                x[0] + v * np.cos(θ) * dt,
                x[1] + v * np.sin(θ) * dt,
                v,
                θ,
                ω
            ])
        
        return np.array([
            x[0] + (v/ω)*(np.sin(θ + ω*dt) - np.sin(θ)),
            x[1] + (v/ω)*(np.cos(θ) - np.cos(θ + ω*dt)),
            v,
            θ + ω*dt,
            ω
        ])

    def jacobian_f(self, x):
        dt = self.ekf.dt
        v, θ, ω = x[2], x[3], x[4]
        
        F = np.eye(5)
        if abs(ω) < 1e-5:  # Jacobian for zero angular velocity
            F[0, 2] = dt * np.cos(θ)
            F[0, 3] = -v * dt * np.sin(θ)
            F[1, 2] = dt * np.sin(θ)
            F[1, 3] = v * dt * np.cos(θ)
        else:  # Jacobian for non-zero angular velocity
            dθ = ω * dt
            F[0, 2] = (np.sin(θ + dθ) - np.sin(θ)) / ω
            F[0, 3] = v/ω * (np.cos(θ + dθ) - np.cos(θ))
            F[0, 4] = (v*dt*np.cos(θ + dθ) - F[0, 2]) / ω
            
            F[1, 2] = (np.cos(θ) - np.cos(θ + dθ)) / ω
            F[1, 3] = v/ω * (np.sin(θ + dθ) - np.sin(θ))
            F[1, 4] = (v*dt*np.sin(θ + dθ) - F[1, 2]) / ω
            
            F[3, 4] = dt
            
        return F

    def predict(self):
        self.ekf.f = self.f
        # Calculate the Jacobian matrix at the current state and assign it to ekf.F
        self.ekf.F = self.jacobian_f(self.ekf.x)
        # Call the predict method without passing the dt parameter
        self.ekf.predict()

    def h(self, x):
        return x[:2]

    def H(self, x):
        return np.array([[1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0]])

    def update(self, z, velocity=None):
        # Dynamically adjust the observation noise based on velocity
        if velocity is not None:
            noise_scale = 1 + 0.1 * velocity  # Higher velocity leads to greater noise
            self.ekf.R = np.array([[0.05 * noise_scale, 0], [0, 0.05 * noise_scale]])
        self.ekf.update(z, HJacobian=self.H, Hx=self.h)

    def get_state(self):
        return self.ekf.x

class KalmanFilterWrapper:
    def __init__(self, dt=1/30.0):
        """
        Initialize the standard Kalman filter.

        :param dt: Time step, default value is 1/30.0.
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # Initial state [x, y, vx, vy]
        self.kf.x = np.array([0., 0., 0., 0.])
        # State transition matrix
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        # Observation matrix
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        # State covariance matrix (initial uncertainty)
        self.kf.P = np.eye(4) * 1000
        # Observation noise
        self.kf.R = np.array([[0.05, 0],
                              [0, 0.05]])
        # Process noise
        self.kf.Q = np.eye(4) * 0.01

    def predict(self):
        """
        Perform the prediction step of the Kalman filter.
        """
        self.kf.predict()

    def update(self, z):
        """
        Perform the update step of the Kalman filter.

        :param z: Observation value.
        """
        self.kf.update(z)

    def get_state(self):
        """
        Get the current state estimate.

        :return: Current state estimate.
        """
        return self.kf.x

class PerformanceEvaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = {
            'lowpass': {'positions': [], 'errors': [], 'jitter': []},
            'ekf': {'positions': [], 'errors': [], 'jitter': []},
            'moving_avg': {'positions': [], 'errors': [], 'jitter': []},
            'kf': {'positions': [], 'errors': [], 'jitter': []}
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