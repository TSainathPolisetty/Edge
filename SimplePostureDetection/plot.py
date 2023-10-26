import pandas as pd
import matplotlib.pyplot as plt

def plot_data_from_csv(filename, label, window_size=10):
    # Read the data from CSV, assuming no header
    data = pd.read_csv(filename, header=None, names=['x', 'y', 'z'])
    
    # Compute rolling mean (moving average) to smooth data
    data_smooth = data.rolling(window=window_size).mean()

    # Create a new figure for each posture
    plt.figure(figsize=(10, 6))
    
    # Plot the smoothed data
    plt.plot(data_smooth['x'], label=f"{label} - x")
    plt.plot(data_smooth['y'], label=f"{label} - y")
    plt.plot(data_smooth['z'], label=f"{label} - z")

    # Setting up the legend, labels, and title
    plt.legend()
    plt.xlabel("Sample Number")
    plt.ylabel("Acceleration (g)")
    plt.title(f"Lying Posture: {label} (Smoothed IMU Data)")
    plt.tight_layout()
    plt.grid(True)
    
    # Display the plot
    plt.show()

# The window size for smoothing
window_size = 10

# Read and plot data from each CSV separately
plot_data_from_csv("smoothed_supine.csv", "Supine", window_size)
plot_data_from_csv("supine.csv", "Supine", window_size)
plot_data_from_csv("smoothed_prone.csv", "Prone", window_size)
plot_data_from_csv("smoothed_sitting.csv", "sitting", window_size)
plot_data_from_csv("smoothed_side.csv", "Side", window_size)
