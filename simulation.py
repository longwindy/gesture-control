# Simulate gesture-controlled mouse and evaluate algorithms
import numpy as np
import matplotlib.pyplot as plt
from algorithm_setting import LowPassFilter, ExtendedKalmanFilterWrapper, MovingAverageFilter, KalmanFilterWrapper, PerformanceEvaluator

# Generate simulation data at 30fps (600 frames = 20 seconds), simulating gesture acceleration and lighting effects
def generate_simulation_data(num_points=300):  # Modified to 30fps
    t = np.linspace(0, 10, num_points)
    
    # Generate a non-linear trajectory (Lissajous curve) with acceleration information added
    acceleration = 0.1 * np.sin(0.2 * t)  # Simulate acceleration
    x_true = 100 * np.sin(0.5 * t) + 50 * np.cos(2.0 * t) + 0.5 * acceleration * t**2
    y_true = 100 * np.cos(0.8 * t) + 50 * np.sin(1.2 * t) + 0.5 * acceleration * t**2
    
    # Dynamic noise (increases with velocity)
    velocity = np.sqrt(np.diff(x_true)**2 + np.diff(y_true)**2)
    velocity = np.append(velocity, velocity[-1])

    # Simulate lighting changes
    light_intensity = 0.7 + 0.3 * np.sin(0.3 * t)  # Light intensity varies between 0.4 and 1.0
    light_factor = np.clip(1.5 - light_intensity, 0.5, 1.5)  # More noise in low light

    noise_std = (1 + 0.1 * velocity) * light_factor
    
    # Simulate pauses (gesture stops)
    pause_mask = np.zeros(num_points, dtype=bool)
    pause_starts = [80, 180, 250]  # Pause frames at 30fps
    for start in pause_starts:
        pause_mask[start:start+15] = True  # 0.5-second pause
    
    x_noisy = x_true.copy()
    y_noisy = y_true.copy()
    for i in range(num_points):
        if not pause_mask[i]:
            x_noisy[i] += np.random.normal(0, noise_std[i])
            y_noisy[i] += np.random.normal(0, noise_std[i])
        else:
            if i > 0:
                x_noisy[i] = x_noisy[i-1]
                y_noisy[i] = y_noisy[i-1]
    
    return x_noisy, y_noisy, light_intensity

# Simulate gesture control
def simulate_gesture_control():
    x_noisy, y_noisy, light_intensity = generate_simulation_data()
    evaluator = PerformanceEvaluator()
    
    # Initialize filters
    lowpass_filter_x = LowPassFilter()
    lowpass_filter_y = LowPassFilter()
    ekf = ExtendedKalmanFilterWrapper(dt=1/30.0)
    moving_avg_filter_x = MovingAverageFilter()
    moving_avg_filter_y = MovingAverageFilter()
    kf = KalmanFilterWrapper(dt=1/30.0)

    # Store trajectories
    lowpass_x, lowpass_y = [], []
    ekf_x, ekf_y = [], []
    moving_avg_x, moving_avg_y = [], []
    kf_x, kf_y = [], []

    for i in range(len(x_noisy)):
        x3, y3 = x_noisy[i], y_noisy[i]

        # Low-pass filter
        lp_x = lowpass_filter_x.filter(x3)
        lp_y = lowpass_filter_y.filter(y3)
        lp_error = np.sqrt((lp_x - x3)**2 + (lp_y - y3)**2)
        evaluator.record('lowpass', (lp_x, lp_y), lp_error)
        lowpass_x.append(lp_x)
        lowpass_y.append(lp_y)

        # Extended Kalman filter
        ekf.predict()
        z = np.array([x3, y3])
        ekf.update(z)
        ekf_state = ekf.get_state()
        ekf_x_val, ekf_y_val = ekf_state[0], ekf_state[1]
        ekf_error = np.sqrt((ekf_x_val - x3)**2 + (ekf_y_val - y3)**2)
        evaluator.record('ekf', (ekf_x_val, ekf_y_val), ekf_error)
        ekf_x.append(ekf_x_val)
        ekf_y.append(ekf_y_val)

        # Moving average
        ma_x = moving_avg_filter_x.filter(x3)
        ma_y = moving_avg_filter_y.filter(y3)
        ma_error = np.sqrt((ma_x - x3)**2 + (ma_y - y3)**2)
        evaluator.record('moving_avg', (ma_x, ma_y), ma_error)
        moving_avg_x.append(ma_x)
        moving_avg_y.append(ma_y)

        # Standard Kalman filter
        kf.predict()
        z_kf = np.array([x3, y3])
        kf.update(z_kf)
        kf_state = kf.get_state()
        kf_x_val, kf_y_val = kf_state[0], kf_state[1]
        kf_error = np.sqrt((kf_x_val - x3)**2 + (kf_y_val - y3)**2)
        evaluator.record('kf', (kf_x_val, kf_y_val), kf_error)
        kf_x.append(kf_x_val)
        kf_y.append(kf_y_val)

    return evaluator, x_noisy, y_noisy, lowpass_x, lowpass_y, ekf_x, ekf_y, moving_avg_x, moving_avg_y, kf_x, kf_y, light_intensity

# Plot performance metrics
def plot_metrics(evaluator):
    metrics = evaluator.get_metrics()
    algorithms = list(metrics.keys())
    avg_errors = [metrics[algo]['avg_error'] for algo in algorithms]
    max_errors = [metrics[algo]['max_error'] for algo in algorithms]
    avg_jitters = [metrics[algo]['avg_jitter'] for algo in algorithms]
    max_jitters = [metrics[algo]['max_jitter'] for algo in algorithms]

    x = np.arange(len(algorithms))
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 1.5 * width, avg_errors, width, label='Avg Error')
    rects2 = ax.bar(x - 0.5 * width, max_errors, width, label='Max Error')
    rects3 = ax.bar(x + 0.5 * width, avg_jitters, width, label='Avg Jitter')
    rects4 = ax.bar(x + 1.5 * width, max_jitters, width, label='Max Jitter')

    ax.set_ylabel('Values')
    ax.set_title('Algorithm Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    fig.tight_layout()
    plt.show()

# Plot trajectory comparison
def plot_trajectories(x_noisy, y_noisy, lowpass_x, lowpass_y, ekf_x, ekf_y, moving_avg_x, moving_avg_y, kf_x, kf_y):
    plt.figure(figsize=(10, 6))
    plt.plot(x_noisy, y_noisy, label='Noisy Data', alpha=0.7)
    plt.plot(lowpass_x, lowpass_y, label='Lowpass Filter', alpha=0.7)
    # Explicitly specify the line color for the Extended Kalman Filter as purple
    plt.plot(ekf_x, ekf_y, label='Extended Kalman Filter', alpha=0.7, color='purple')
    plt.plot(moving_avg_x, moving_avg_y, label='Moving Average', alpha=0.7)
    plt.plot(kf_x, kf_y, label='Kalman Filter', alpha=0.7)
    plt.title('Trajectory Comparison')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot light intensity variation
def plot_light_intensity(light_intensity):
    t = np.linspace(0, 10, len(light_intensity))
    plt.figure(figsize=(10, 6))
    plt.plot(t, light_intensity, label='Light Intensity')
    plt.title('Light Intensity Variation')
    plt.xlabel('Time (s)')
    plt.ylabel('Light Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluator, x_noisy, y_noisy, lowpass_x, lowpass_y, ekf_x, ekf_y, moving_avg_x, moving_avg_y, kf_x, kf_y, light_intensity = simulate_gesture_control()
    plot_metrics(evaluator)
    plot_trajectories(x_noisy, y_noisy, lowpass_x, lowpass_y, ekf_x, ekf_y, moving_avg_x, moving_avg_y, kf_x, kf_y)
    plot_light_intensity(light_intensity)