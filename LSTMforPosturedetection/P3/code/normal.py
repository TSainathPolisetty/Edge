import pandas as pd
import os

def mean_normalize(df):
    return (df - df.mean()) / (df.max() - df.min())

def process_file(filename, output_directory):
    # Read the labeled data
    df = pd.read_csv(filename)

    # Extract the features and labels
    features = df.iloc[:, :-1]  # Assuming label is the last column
    labels = df.iloc[:, -1]

    # Normalize the features
    normalized_features = mean_normalize(features)

    # Concatenate normalized features with labels
    normalized_df = pd.concat([normalized_features, labels], axis=1)

    # Save the normalized data
    base_name = os.path.basename(filename)
    normalized_filename = os.path.join(output_directory, f"normalized_{base_name}")
    normalized_df.to_csv(normalized_filename, index=False)

# List of labeled files
filenames = ["prone_labeled.csv", "supine_labeled.csv", "side_labeled.csv", "sitting_labeled.csv", "prone_windowed_labeled.csv", "supine_windowed_labeled.csv", "side_windowed_labeled.csv", "sitting_windowed_labeled.csv", "prone_baseline_labeled.csv", "supine_baseline_labeled.csv", "side_baseline_labeled.csv", "sitting_baseline_labeled.csv", "prone_baseline_windowed_labeled.csv", "supine_baseline_windowed_labeled.csv", "side_baseline_windowed_labeled.csv", "sitting_baseline_windowed_labeled.csv"]
output_directory = "/home/tulasi/eml/P3/"


for filename in filenames:
    process_file(filename, output_directory)
