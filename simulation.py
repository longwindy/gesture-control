import numpy as np
import matplotlib.pyplot as plt
from algorithm_setting import lowpass_filter, init_kalman_filter, moving_average, PerformanceEvaluator

# Generate more realistic simulated gesture data
def generate_simulation_data(num_points=1000):
    t = np.linspace(0, 10, num_points)
    
    # Generate more complex true trajectory
    x_true = 100 * np.sin(0.5 * t) + 50 * np.cos(1.5 * t)
    y_true = 100 * np.cos(0.5 * t) + 50 * np.sin(1.5 * t)
    
    # Introduce dynamic noise
    noise_std = 2 + 3 * np.sin(0.2 * t)
    
    # Simulate pauses in gesture movement
    pause_mask = np.zeros(num_points, dtype=bool)
    pause_length = 20
    pause_start = [200, 500, 800]
    for start in pause_start:
        pause_mask[start:start + pause_length] = True
    
    x_noisy = x_true.copy()
    y_noisy = y_true.copy()
    for i in range(num_points):
        if not pause_mask[i]:
            x_noisy[i] += np.random.normal(0, noise_std[i])
            y_noisy[i] += np.random.normal(0, noise_std[i])
        else:
            x_noisy[i] = x_noisy[i - 1] if i > 0 else x_true[i]
            y_noisy[i] = y_noisy[i - 1] if i > 0 else y_true[i]
    
    return x_noisy, y_noisy

# Simulate gesture-controlled mouse and evaluate algorithms
def simulate_gesture_control():
    x_noisy, y_noisy = generate_simulation_data()
    evaluator = PerformanceEvaluator()
    kf = init_kalman_filter()
    ma_history_x = []
    ma_history_y = []
    pLocx, pLocy = 0, 0

    # Store trajectory data
    lowpass_x = []
    lowpass_y = []
    kalman_x = []
    kalman_y = []
    moving_avg_x = []
    moving_avg_y = []

    for i in range(len(x_noisy)):
        x3, y3 = x_noisy[i], y_noisy[i]

        # Low-pass filtering
        x_data = np.array([pLocx, x3])
        y_data = np.array([pLocy, y3])
        lp_x, lp_y = lowpass_filter(x_data)[-1], lowpass_filter(y_data)[-1]
        lp_error = np.sqrt((lp_x - x3)**2 + (lp_y - y3)**2)
        evaluator.record('lowpass', (lp_x, lp_y), lp_error)
        lowpass_x.append(lp_x)
        lowpass_y.append(lp_y)

        # Kalman filtering
        z = np.array([[x3], [y3]])
        kf.predict()
        kf.update(z)
        kf_x, kf_y = kf.x[0], kf.x[1]
        kf_error = np.sqrt((kf_x - x3)**2 + (kf_y - y3)**2)
        evaluator.record('kalman', (kf_x, kf_y), kf_error)
        kalman_x.append(kf_x)
        kalman_y.append(kf_y)

        # Moving average
        ma_x = moving_average(ma_history_x, x3)
        ma_y = moving_average(ma_history_y, y3)
        ma_error = np.sqrt((ma_x - x3)**2 + (ma_y - y3)**2)
        evaluator.record('moving_avg', (ma_x, ma_y), ma_error)
        moving_avg_x.append(ma_x)
        moving_avg_y.append(ma_y)

        pLocx, pLocy = kf_x, kf_y

    return evaluator, x_noisy, y_noisy, lowpass_x, lowpass_y, kalman_x, kalman_y, moving_avg_x, moving_avg_y

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
def plot_trajectories(x_noisy, y_noisy, lowpass_x, lowpass_y, kalman_x, kalman_y, moving_avg_x, moving_avg_y):
    plt.figure(figsize=(10, 6))
    plt.plot(x_noisy, y_noisy, label='Noisy Data', alpha=0.7)
    plt.plot(lowpass_x, lowpass_y, label='Lowpass Filter', alpha=0.7)
    plt.plot(kalman_x, kalman_y, label='Kalman Filter', alpha=0.7)
    plt.plot(moving_avg_x, moving_avg_y, label='Moving Average', alpha=0.7)
    plt.title('Trajectory Comparison')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluator, x_noisy, y_noisy, lowpass_x, lowpass_y, kalman_x, kalman_y, moving_avg_x, moving_avg_y = simulate_gesture_control()
    plot_metrics(evaluator)
    plot_trajectories(x_noisy, y_noisy, lowpass_x, lowpass_y, kalman_x, kalman_y, moving_avg_x, moving_avg_y)