import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming normalized files are prefixed with 'normalized_'
normalized_files = ["normalized_prone_labeled.csv","normalized_supine_labeled.csv","normalized_side_labeled.csv", "normalized_sitting_labeled.csv"]

# Combine all the files into a single dataframe
all_data = pd.concat([pd.read_csv(file) for file in normalized_files], ignore_index=True)

# Split the data
train, temp = train_test_split(all_data, test_size=0.3, random_state=42)
validate, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save the split datasets
train.to_csv("train_data.csv", index=False)
validate.to_csv("validate_data.csv", index=False)
test.to_csv("test_data.csv", index=False)
