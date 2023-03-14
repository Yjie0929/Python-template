# Import the required packages and classes
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing  # Import the dataset fetcher
from sklearn.linear_model import LinearRegression  # Import the linear regression model
from sklearn.metrics import mean_squared_error  # Import the mean squared error metric
import matplotlib.pyplot as plt  # Import matplotlib

# Fetch the data and convert it to DataFrame format
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target)

# Build a linear regression model and fit the data
lr = LinearRegression()
lr.fit(X, y)

# Print the model parameters (coefficients and intercept)
print("Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)

# Predict the house values on the whole dataset and calculate the mean squared error
y_pred = lr.predict(X)
mse = mean_squared_error(y, y_pred)
print("Mean squared error:", mse)

# Plot a scatter plot and a regression line (using MedInc (median income) as an independent variable)
plt.scatter(X['MedInc'], y, color='blue', label='Actual')  # Plot a scatter plot with blue color and label 'Actual'
plt.scatter(X['MedInc'], y_pred, color='red', label='Predicted')  
# Plot a scatter plot with red color and label 'Predicted'

m, b = np.polyfit(X['MedInc'], y_pred, 1)  # Calculate the slope and intercept of the regression line

plt.plot(X['MedInc'], m * X['MedInc'] + b, color='black', label='Regression line')  
# Plot a regression line with black color and label 'Regression line'

plt.xlabel('Median income in block')  # Set x-axis label
plt.ylabel('Median house value in block')  # Set y-axis label

plt.legend()  # Show legend

plt.show()  # Show figure
