#Import scikit-learn dataset library
from sklearn.model_selection import train_test_split
from sklearn import metrics

import os
# get max number of jobs to run (leave 2 cores for OS etc.)
N_CORES = max(1, os.cpu_count() - 2) # -2 CPU cores for OS etc.

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

from csv_to_bunch import parse_csv

training_file = "datasets/TrainingDataBinary.csv"
testing_file = "datasets/TestingDataBinary.csv"

# Load dataset
binary = parse_csv(training_file)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(binary.data, binary.target, test_size=0.3) # 70% training and 30% test

# Define a Standard Scaler to normalize inputs
scaler = StandardScaler()

# Define a PCA (Principal component analysis) to reduce the dimensionality of the data
pca = PCA()

# Define a MLP (Multi-layer Perceptron) classifier to classify the data
mlp = MLPClassifier(
    solver='adam',
    learning_rate="adaptive",
    max_iter=10000,
    activation="tanh",
    alpha=1e-7,
    beta_1=0.999,
    beta_2=0.99999,
    epsilon=1e-6,
    hidden_layer_sizes=(128*2, 128)
)

# Define the search grid
param_grid = {
# Uncomment the following lines to add parameters to the search grid
    # "activation": ["identity", "logistic", "tanh", "relu"],
    # "alpha": np.logspace(-7, -4, 4),
    # "beta_1": [0.9, 0.99, 0.999],
    # "beta_2": [0.999, 0.9999, 0.99999],
    # "epsilon": [1e-7, 1e-8, 1e-9],
    # "epsilon": [1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
}
search = GridSearchCV(mlp, param_grid, n_jobs=N_CORES, cv=5)

# Add scaler+PCA outside the search to avoid it getting unnecessarily re-run
pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("search", search)])

# Train the model using the training sets
pipe.fit(X_train, y_train)

# Print out the best parameters
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

# Predict the response for test dataset
y_pred = pipe.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
