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

import pandas as pd
from csv_to_bunch import parse_csv

training_file = "TrainingDataBinary.csv"
testing_file = "TestingDataBinary.csv"

#Load dataset
# binary = pd.read_csv(training_file)

binary = parse_csv(training_file)

# print the names of the  features
print("Features: ", binary.feature_names)

# print the label type of data
print("Labels: ", binary.target_names)

# # print data(feature)shape
print(binary.data.shape)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(binary.data, binary.target, test_size=0.3) # 70% training and 30% test


# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
pca = PCA()
# Define a Standard Scaler to normalize inputs
scaler = StandardScaler()

# set the tolerance to a large value to make the example faster
# logistic = LogisticRegression(max_iter=1000000, tol=0.05)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

tree = DecisionTreeClassifier(criterion="log_loss")

mlp = MLPClassifier(solver='adam', max_iter=10000, alpha=1e-5, learning_rate="adaptive", activation="logistic")

# pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logistic)])
# pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("tree", tree)])
pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("mlp", mlp)])
param_grid = {
    # "pca__n_components": list(range(5, 125, 10)),
    # "pca__n_components": list(range(75, 105, 5)),
    # "pca__n_components": list(range(78, 82, 1)),
    "pca__n_components": list(range(90, 100, 2)),
    # "logistic__C": np.logspace(-1, 5, 1+5+1),
    # "logistic__C": np.logspace(2, 5, -2+5+1),
    # "mlp__activation": ["identity", "logistic", "tanh", "relu"],
}
search = GridSearchCV(pipe, param_grid, n_jobs=N_CORES, cv=5)


#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
search.fit(X_train, y_train)

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#Predict the response for test dataset
y_pred = search.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

testing_data = parse_csv(testing_file, has_labels=False)
testing_pred = search.predict(testing_data)

print(testing_pred)