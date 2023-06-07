import pandas as pd
import numpy as np
from sklearn.utils import Bunch
import csv

# this parses a CSV file into a sklearn Bunch in a similar format to those found in sklearn.datasets
def parse_csv(csv_file, has_labels=True, label_name="marker"):
    # read in data from CSV
    df = pd.read_csv(csv_file)
    if not has_labels:
        # if data has no labels, just output an numpy array of the data
        return df.to_numpy()

    # get feature names and remove label from features
    feature_names = list(df.columns.values)
    assert label_name in feature_names, f"label {label_name} not found"
    label_index = feature_names.index(label_name)
    feature_names.remove(label_name)

    # get data and seperate the labels from the data
    data = df.to_numpy()
    target = data[:, label_index]
    data = np.delete(data, label_index, 1) # remove labels from data
    
    # get list of possible labels
    target_names = list(set(target))

    # simple data container used by sklearn (basically a dictionary), allowing access by attribute
    return Bunch(
        data=data,
        target=target,
        feature_names=feature_names,
        target_names=target_names,
        DESCR=f"CSV data from {csv_file}",
        filename=csv_file
    )

integer_columns = ["control_panel_log1", "control_panel_log2", "control_panel_log3", "control_panel_log4", "relay1_log", "relay2_log", "relay3_log", "relay4_log", "snort_log1", "snort_log2", "snort_log3", "snort_log4", "marker"]
def save_in_training_format(data, labels, test_csv_file, training_csv_file, output_csv_file, label_name="marker"):
    # get column labels from training and test files
    test_cols = list(pd.read_csv(test_csv_file).columns.values)
    train_cols = list(pd.read_csv(training_csv_file).columns.values)

    assert label_name in train_cols, f"label {label_name} not found"
    
    with open(output_csv_file, "w", newline="") as f:
        # use field names from training file
        writer = csv.DictWriter(f, fieldnames=train_cols)
        writer.writeheader()

        for row, label in zip(data, labels):
            # create dict of test data, and add label column
            row_data = {col: row[i] for i, col in enumerate(test_cols)}
            row_data[label_name] = label

            # convert columns that should be integers to integers
            for col in integer_columns:
                if col in row_data:
                    row_data[col] = int(row_data[col])
                    
            # write data to csv file
            writer.writerow(row_data)

    print(f"Data saved as {output_csv_file}")
