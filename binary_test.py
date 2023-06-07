import joblib
from csv_to_bunch import parse_csv, save_in_training_format


training_file = "datasets/TrainingDataBinary.csv"
testing_file = "datasets/TestingDataBinary.csv"
output_file = "output/TestingDataBinaryLabelled.csv"

# load the model
model = joblib.load("models/binary.model")

# load the testing dataset
testing_data = parse_csv(testing_file, has_labels=False)

# predict the labels
testing_pred = model.predict(testing_data)

# save the labelled data
save_in_training_format(testing_data, testing_pred, testing_file, training_file, output_file)
