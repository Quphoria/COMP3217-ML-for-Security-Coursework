#Import scikit-learn dataset library
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from csv_to_bunch import parse_csv

training_file = "datasets/TrainingDataMulti.csv"

#Load dataset
binary = parse_csv(training_file)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(binary.data, binary.target, test_size=0.2) # 80% training and 20% test

# Define a Standard Scaler to normalize inputs
scaler = StandardScaler()

# Define a PCA (Principal component analysis) to reduce the dimensionality of the data
pca = PCA()

# Define a MLP (Multi-layer Perceptron) classifier to classify the data
mlp = MLPClassifier(
    solver='adam',
    learning_rate="adaptive",
    max_iter=20000,
    activation="tanh",
    alpha=1e-5,
    beta_1=0.99,
    beta_2=0.9999,
    epsilon=1e-6,
    hidden_layer_sizes=(128*4, 128*2, 128)
)

pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("mlp", mlp)])

#Train the model using the training sets
pipe.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = pipe.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# save pipe
joblib.dump(pipe, "models/multi.model")
with open("models/multi.model.stats", "w") as f:
    f.write(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")