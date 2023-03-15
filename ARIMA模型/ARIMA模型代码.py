# Import necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

# Generate a random time series dataset
np.random.seed(123)
data = np.random.normal(0, 1, 1000)
data = pd.Series(data)

# Plot the data graph
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Random Time Series Data')
plt.show()

# Define and fit an ARIMA(2,1,2) model
model = ARIMA(data, order=(2,1,2))
model_fit = model.fit()

# Print model summary and parameters
print(model_fit.summary())
print(model_fit.params)

# Plot the residual graph
residuals = pd.DataFrame(model_fit.resid)
plt.figure(figsize=(10, 6))
residuals.plot()
plt.title('Residuals')
plt.show()

# Calculate the mean squared error (MSE) of the residuals
mse = np.mean(residuals**2)
print('MSE: ', mse)

# Predict the values for the next 10 time points
forecast = model_fit.forecast(steps=10)[0]
print('Forecast: ', forecast)

# Plot the predicted values graph
plt.figure(figsize=(10, 6))
plt.plot(data.index[-50:], data.values[-50:], label='Original')
plt.plot(np.arange(1000, 1010), forecast, label='Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Forecast using ARIMA(2,1,2)')
plt.legend()
plt.show()
