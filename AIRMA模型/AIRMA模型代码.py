# 导入必要的库
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

# 生成一个随机时间序列数据集
np.random.seed(123)
data = np.random.normal(0, 1, 1000)
data = pd.Series(data)

# 绘制数据图像
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Random Time Series Data')
plt.show()

# 定义并拟合ARIMA(2,1,2)模型
model = ARIMA(data, order=(2,1,2))
model_fit = model.fit()

# 打印模型摘要和参数
print(model_fit.summary())
print(model_fit.params)

# 绘制残差图像
residuals = pd.DataFrame(model_fit.resid)
plt.figure(figsize=(10, 6))
residuals.plot()
plt.title('Residuals')
plt.show()

# 计算残差的均方误差（MSE）
mse = np.mean(residuals**2)
print('MSE: ', mse)

# 预测未来10个时间点的值
forecast = model_fit.forecast(steps=10)[0]
print('Forecast: ', forecast)

# 绘制预测图像
plt.figure(figsize=(10, 6))
plt.plot(data.index[-50:], data.values[-50:], label='Original')
plt.plot(np.arange(1000, 1010), forecast, label='Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Forecast using ARIMA(2,1,2)')
plt.legend()
plt.show()
