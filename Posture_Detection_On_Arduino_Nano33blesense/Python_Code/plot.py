import pandas as pd
import matplotlib.pyplot as plt

def plot_data_from_csv(filename, label, window_size=10):
    # Read the data from CSV
    column_names = ['Timestamp', 'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'MagX', 'MagY', 'MagZ', 'Posture']
    data = pd.read_csv(filename, skiprows=1, header=None, names=column_names)

    
    # Exclude non-numeric columns
    numeric_data = data.drop(columns=['Timestamp', 'Posture'])
    numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce').dropna()


    # Compute rolling mean (moving average) to smooth data
    data_smooth = numeric_data.rolling(window=window_size).mean()

    sensors = ['Accel', 'Gyro', 'Mag']

    for sensor in sensors:
        # Create a new figure for each posture and sensor
        plt.figure(figsize=(10, 6))
        
        # Plot the smoothed data
        plt.plot(data_smooth[f'{sensor}X'], label=f"{sensor} X")
        plt.plot(data_smooth[f'{sensor}Y'], label=f"{sensor} Y")
        plt.plot(data_smooth[f'{sensor}Z'], label=f"{sensor} Z")

        # Setting up the legend, labels, and title
        plt.legend()
        plt.xlabel("Sample Number")
        plt.ylabel(f"{sensor} Readings")
        plt.title(f"Lying Posture: {label} (Smoothed {sensor} Data)")
        plt.tight_layout()
        plt.grid(True)
        
        # Display the plot
        plt.show()

# The window size for smoothing
window_size = 10

# Read and plot data from the CSV
plot_data_from_csv("supine.csv", "Supine", window_size)
plot_data_from_csv("prone.csv", "Prone", window_size)
plot_data_from_csv("side.csv", "Side", window_size)
plot_data_from_csv("sitting.csv", "Sitting", window_size)



