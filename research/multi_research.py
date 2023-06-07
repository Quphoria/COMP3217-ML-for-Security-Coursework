#Import scikit-learn dataset library
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics

import numpy as np
import os
N_CORES = max(1, os.cpu_count() - 2) # -2 CPU cores for OS etc.

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
from csv_to_bunch import parse_csv

training_file = "TrainingDataMulti.csv"
testing_file = "TestingDataMulti.csv"

#Load dataset
# binary = pd.read_csv(training_file)

multi = parse_csv(training_file)

# print the names of the  features
print("Features: ", multi.feature_names)

# print the label type of data
print("Labels: ", multi.target_names)

# ohe = OneHotEncoder()
# transformed_labels = ohe.fit_transform(multi.target.reshape(-1, 1)).toarray() # single feature so reshape to 2d

# # print data(feature)shape
print(multi.data.shape)
# print(multi.target.shape, transformed_labels.shape)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(multi.data, multi.target, test_size=0.2) # 70% training and 30% test


# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
pca = PCA(n_components=100)
# Define a Standard Scaler to normalize inputs
scaler = StandardScaler()

# set the tolerance to a large value to make the example faster
# logistic = LogisticRegression(max_iter=1000000, tol=0.05)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

tree = DecisionTreeClassifier(criterion="log_loss")

mlp = MLPClassifier(solver='adam', max_iter=20000, beta_1=0.9, beta_2=0.99999, alpha=1e-7, epsilon=1e-8, learning_rate="adaptive", activation="tanh", hidden_layer_sizes=(128*4, 128*2, 128))

# pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logistic)])
# pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("tree", tree)])
pipe = Pipeline(steps=[("pca", pca), ("mlp", mlp)])
param_grid = {
    # "pca__n_components": list(range(5, 125, 5)),
    "mlp__activation": ["identity", "logistic", "tanh", "relu"],
    "mlp__beta_1": [0.9, 0.99, 0.999],
    "mlp__beta_2": [0.999, 0.9999, 0.99999],
    "mlp__epsilon": [1e-7, 1e-8, 1e-9],
    # "mlp__alpha": np.logspace(-7, -4, 4),
    # "pca__n_components": list(range(70, 105, 5)),
    # "pca__n_components": list(range(95, 105, 1)),
}
search = GridSearchCV(pipe, param_grid, n_jobs=N_CORES, cv=5)

full_pipe = Pipeline(steps=[("scaler", scaler), ("search", search)])

#Train the model using the training sets
full_pipe.fit(X_train, y_train)

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#Predict the response for test dataset
y_pred = full_pipe.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

testing_data = parse_csv(testing_file, has_labels=False)
testing_pred = full_pipe.predict(testing_data)

print(testing_pred)
