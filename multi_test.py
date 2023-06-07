import joblib
from csv_to_bunch import parse_csv, save_in_training_format


training_file = "datasets/TrainingDataMulti.csv"
testing_file = "datasets/TestingDataMulti.csv"
output_file = "output/TestingDataMultiLabelled.csv"

# load the model
model = joblib.load("models/multi.model")

# load the testing dataset
testing_data = parse_csv(testing_file, has_labels=False)

# predict the labels
testing_pred = model.predict(testing_data)

# save the labelled data
save_in_training_format(testing_data, testing_pred, testing_file, training_file, output_file)
