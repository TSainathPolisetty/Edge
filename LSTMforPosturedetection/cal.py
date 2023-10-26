import pandas as pd

# Load the CSV file into a DataFrame
file_path = "train_data.csv"
df = pd.read_csv(file_path)

# Calculate min, max, and mean for each column
min_values = df.min(numeric_only=True)
max_values = df.max(numeric_only=True)
mean_values = df.mean(numeric_only=True)

# Print the results
print("Minimum values:\n", min_values)
print("\nMaximum values:\n", max_values)
print("\nMean values:\n", mean_values)
