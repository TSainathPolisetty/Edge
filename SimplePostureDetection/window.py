import pandas as pd
import os

def read_csv_files(filenames):
    data_frames = []
    for filename in filenames:
        df = pd.read_csv(filename)
        data_frames.append((df, filename))
    return data_frames

def windowed_average(dataframe, window_size, step_size):
    """
    Returns a list of average values for each window.
    
    :param dataframe: DataFrame containing the data.
    :param window_size: Size of each window (number of samples).
    :param step_size: Number of data points to step for the next window.
    :return: List of DataFrames with averaged values.
    """
    windows_avg = []
    start = 0
    previous_avg = dataframe.iloc[0]  # To keep track of the most recent average value

    while start < len(dataframe):
        end = start + window_size
        if end <= len(dataframe):
            current_window = dataframe.iloc[start:end]
            current_avg = current_window.mean()
            
            # If any column lacks data in the current window, use the most recent value
            for col in current_avg.index:
                if pd.isna(current_avg[col]):
                    current_avg[col] = previous_avg[col]
            windows_avg.append(current_avg)
            previous_avg = current_avg
        
        start += step_size

    # Convert list of Series to DataFrame
    return pd.concat(windows_avg, axis=1).transpose()

filenames = ["prone_baseline.csv", "supine_baseline.csv", "side_baseline.csv", "sitting_baseline.csv"]
dataframes = read_csv_files(filenames)

window_size = 20  # 2 seconds * 10Hz
step_size = 10    # 50% overlap

windowed_avg_data = []
for dataframe, filename in dataframes:
    avg_df = windowed_average(dataframe, window_size, step_size)
    windowed_avg_data.append((avg_df, filename))

# Save windowed average data with modified filenames:
for avg_df, original_filename in windowed_avg_data:
    new_filename = os.path.splitext(original_filename)[0] + "_windowed.csv"
    avg_df.to_csv(new_filename, index=False)
