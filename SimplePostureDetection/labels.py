import pandas as pd

def label_data(filename):
    if 'supine' in filename:
        return 1
    elif 'prone' in filename:
        return 2
    elif 'side' in filename:
        return 3
    elif 'sitting' in filename:
        return 4
    else:
        return 5

def process_and_save(filename):
    # Read the CSV without a header
    df = pd.read_csv(filename, header=None)
    
    # Assign labels based on the filename
    label = label_data(filename)
    df['label'] = label
    
    # Rename columns
    df.columns = ['x', 'y', 'z', 'label']
    
    # Construct the output filename
    output_filename = filename.split(".")[0] + "_labeled.csv"
    
    # Save the updated dataframe
    df.to_csv(output_filename, index=False)

# List of your files
filenames = ["prone.csv", "supine.csv", "side.csv", "sitting.csv", "prone_windowed.csv", "supine_windowed.csv", "side_windowed.csv", "sitting_windowed.csv"]

for filename in filenames:
    process_and_save(filename)
