# Import necessary libraries
import numpy as np # Mathematical operations library
import pandas as pd # Data processing library
import matplotlib.pyplot as plt # Data visualization library
from sklearn.datasets import load_breast_cancer # Breast cancer dataset
from sklearn.linear_model import LogisticRegression # Logistic regression class
from sklearn.metrics import accuracy_score, confusion_matrix # Model evaluation metrics
from sklearn.preprocessing import StandardScaler # Data standardization class

# Load the dataset and check basic information
data = load_breast_cancer() # Return a dictionary object containing data and meta-information
X = data['data'] # Feature matrix X (569*30)
y = data['target'] # Target variable y (569*1), 0 means malignant, 1 means benign
feature_names = data['feature_names'] # Feature names (30)
target_names = data['target_names'] # Target names (2)
print('Data shape:', X.shape)
print('Feature names:', feature_names)
print('Target names:', target_names)

# Visualize part of the dataset (only select the first two features for plotting)
plt.scatter(X[y==0, 0], X[y==0, 1], c='b', label=target_names[0])
plt.scatter(X[y==1, 0], X[y==1, 1], c='r', label=target_names[1])
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.show()

# Split the dataset into training set and test set (ratio of 8:2)
np.random.seed(0) # Set random seed for reproducibility of results
indices = np.random.permutation(len(X)) # Randomly shuffle index values
train_size = int(len(X) * 0.8) # Training set size (80%)
X_train = X[indices[:train_size]] # Training set feature matrix (455*30)
y_train = y[indices[:train_size]] # Training set target variable (455*1)
X_test = X[indices[train_size:]] # Test set feature matrix (114*30)
y_test = y[indices[train_size:]] # Test set target variable (114*1)

# Standardize the data to make each feature have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use the LogisticRegression class in sklearn.linear_model to fit and predict the model, and increase the number of iterations max_iter to 1000, choose solver as 'newton-cg'
lr = LogisticRegression(max_iter=1000, solver='newton-cg')
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

# Calculate the accuracy and confusion matrix of the model and print the results
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print('Accuracy:', acc)
print('Confusion matrix:\n', cm)
