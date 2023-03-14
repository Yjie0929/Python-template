# import numpy as np  # for scientific computing
# import pandas as pd  # for data processing and analysis
import matplotlib.pyplot as plt  # for data visualization
from sklearn.datasets import load_breast_cancer  # breast cancer dataset
from sklearn.linear_model import LogisticRegression  # logistic regression model
from sklearn.metrics import accuracy_score  # accuracy evaluation

# Load the dataset and split it into training set and test set
X, y = load_breast_cancer(return_X_y=True)  # X is the feature matrix, y is the label vector
X_train = X[:400]  # take the first 400 samples as the features of the training set
y_train = y[:400]  # take the first 400 samples as the labels of the training set
X_test = X[400:]  # take the last 169 samples as the features of the test set
y_test = y[400:]  # take the last 169 samples as the labels of the test set

# Create and train a logistic regression model
model = LogisticRegression(max_iter=500, solver="newton-cg")  
# create a logistic regression object, you can specify parameters such as regularization term, learning rate etc., default using L2 regularization and liblinear solver.
model.fit(X_train, y_train)  
# use the features and labels of the training set to fit the model, i.e. solve for model parameters.

# Predict on test set and calculate accuracy score
y_pred = model.predict(X_test)  # use model to predict on test set features, get prediction results.
acc = accuracy_score(y_test, y_pred)  
# use accuracy_score function to calculate accuracy score between prediction results and true labels, i.e. proportion of correct predictions.
print("Accuracy score:", acc)  # print accuracy score.

# Plot comparison between prediction results and true labels
plt.scatter(range(len(y_test)), y_test, label="True labels")  
# use scatter function to plot scatter plot, x-axis is sample index number, y-axis is true label (0 or 1), and set legend.
plt.scatter(range(len(y_pred)), y_pred, label="Prediction results")  
# same as above, y-axis is prediction result (0 or 1).
plt.ylim(0, 2) # set the display range of Y axis to 0-10
plt.legend()  # show legend.
plt.show()  # show image.
