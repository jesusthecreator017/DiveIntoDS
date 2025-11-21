"""
Cleaning Erroneous Data
"""
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data into memory
carSales = pd.read_csv('carsales.csv')

# Peek at the data
print(f"First Five Rows:\n{carSales.head()}")

# Reformat the column names
carSales.columns = ['month', 'sales']

# Peek at the bottom of the data (We can see the last row containes a NaN
print(f"Last Five Rows:\n{carSales.tail()}\n")

carSales = carSales.loc[:107, :].copy()

print(f"Last Five Rows:\n{carSales.tail()}\n")

# Create a period column
carSales['period'] = list(range(108))

print(f"Last Five Rows:\n{carSales.tail()}")

"""
Plotting Data to Find Trends
"""
# Display the plot
plt.scatter(carSales['period'], carSales['sales'])
plt.title('Car Sales by Month')
plt.xlabel('Period')
plt.ylabel('Sales')
plt.show()

"""
Performing Linear Regression
"""

x = carSales['period'].values.reshape(-1, 1)
y = carSales['sales'].values.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(x, y)

print(f"\nRegressor Coefficient: {regressor.coef_}")
print(f"Regressor Intercept: {regressor.intercept_}")

# Create a plot with the lines rendered in
plt.scatter(carSales['period'], carSales['sales'])
plt.plot(carSales['period'], [81.2 * i + 10250.8 for i in carSales['period']], 'r-', label='Regression Line')
plt.plot(carSales['period'], [125 * i + 8000 for i in carSales['period']], 'r--', label='Hypothesized Line')
plt.legend(loc='upper left')
plt.title('Car Sales by Month')
plt.xlabel('Period')
plt.ylabel('Sales')
plt.show()

"""
Calculating Error Measurement
"""

salesList = carSales['sales'].tolist()
regressorLine = [81.2 * i + 10250.8 for i in carSales['period']]
hypothesizedLine = [125 * i + 8000 for i in carSales['period']]
error_1 = [(x - y) for x, y in zip(regressorLine, salesList)]
error_2 = [(x - y) for x, y in zip(hypothesizedLine, salesList)]

error1_abs = [abs(value) for value in error_1]
error2_abs = [abs(value) for value in error_2]

print(f"Mean Absolute Error 1: {np.mean(error1_abs)}")
print(f"Mean Absolute Error 2: {np.mean(error2_abs)}")