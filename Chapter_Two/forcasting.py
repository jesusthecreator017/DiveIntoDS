"""
Cleaning Erroneous Data
"""
# Import Libraries
import math
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

error1_squared = [value**2 for value in error_1]
error2_squared = [value**2 for value in error_2]

print(f"Mean Squared Error 1: {np.sqrt(np.mean(error1_squared))}")
print(f"Mean Squared Error 2: {np.sqrt(np.mean(error2_squared))}")

def get_mae(line, actual):
    error = [x - y for x, y in zip(line, actual)]
    errorAbs = [abs(value) for value in error]
    mae = np.mean(errorAbs)
    return mae

def get_rmse(line, actual):
    error = [x - y for x, y in zip(line, actual)]
    error_squared = [value**2 for value in error]
    rmse = np.sqrt(np.mean(error_squared))
    return rmse

"""
Using Regression to Forcast Future Trends
"""

x_extended = np.append(carSales['period'], np.arange(108, 116))
x_extended = x_extended.reshape(-1, 1)
extended_prediction = regressor.predict(x_extended)

plt.scatter(carSales['period'], carSales['sales'])
plt.plot(x_extended, extended_prediction, 'r--')
plt.title('Car Sales by Month')
plt.xlabel('Period')
plt.ylabel('Sales')
plt.show()

"""
Trying More Regression Models
"""

carSales['quadratic'] = carSales['period'].apply(lambda x: x ** 2)
carSales['cubic'] = carSales['period'].apply(lambda x: x ** 3)

x3 = carSales.loc[:, ['period', 'quadratic', 'cubic']].values.reshape(-1, 3)
y = carSales['sales'].values.reshape(-1, 1)

regressor_cubic = LinearRegression()
regressor_cubic.fit(x3, y)

plt.scatter(carSales['period'], carSales['sales'])
plt.plot(x, regressor.predict(x), 'r-')
plt.plot(x, regressor_cubic.predict(x3), 'r--')
plt.title('Car Sales by Month')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

print(f"\nCubic Regressor Coefficients: {regressor_cubic.coef_}")
print(f"Cubic Regressor Intercept: {regressor_cubic.intercept_}")

"""
Trigonometry to Capture Variations
"""

plt.plot(carSales['period'], carSales['sales'])
plt.title('Car Sales by Month')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

carSales['sin_period'] = carSales['period'].apply(lambda x: math.sin(x * 2 * math.pi / 12))
carSales['cos_period'] = carSales['period'].apply(lambda x: math.cos(x * 2 * math.pi / 12))

x_trig = carSales.loc[:, ['period', 'sin_period', 'cos_period']].values.reshape(-1, 3)
y = carSales['sales'].values.reshape(-1, 1)

regressor_trig = LinearRegression()
regressor_trig.fit(x_trig, y)

plt.plot(carSales['period'], carSales['sales'])
plt.plot(x, regressor_trig.predict(x_trig), 'r--')
plt.title('Car Sales by Month')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

trigLine = regressor_trig.predict(x_trig)[:, 0]
print(f"Trig Regressor RMSE: {get_rmse(trigLine, salesList)}")

"""
Choosing the Best Regression to Use for Forecasting
"""

carSales['squareroot'] = carSales['period'].apply(lambda x: np.sqrt(x))
carSales['exponent15'] = carSales['period'].apply(lambda x: x ** 1.5)
carSales['log'] = carSales['period'].apply(lambda x: math.log(x + 1))

x_complex = carSales.loc[:, ['period', 'log', 'sin_period', 'cos_period', 'squareroot', 'exponent15', 'log', 'quadratic', 'cubic']].values.reshape(-1, 9)
y = carSales['sales'].values.reshape(-1, 1)

regressor_complex = LinearRegression()
regressor_complex.fit(x_complex, y)

complex_line = [prediction for sublist in regressor_complex.predict(x_complex) for prediction in sublist]
print(f"Complex Line RMSE: {get_rmse(complex_line, salesList)}")

x_complex_train = carSales.loc[0:80, ['period', 'log', 'sin_period', 'cos_period', 'squareroot', 'exponent15', 'log', 'quadratic', 'cubic']].values.reshape(-1, 9)
y_train = carSales.loc[0:80, 'sales'].values.reshape(-1, 1)

x_complex_test = carSales.loc[81:107, ['period', 'log', 'sin_period', 'cos_period', 'squareroot', 'exponent15', 'log', 'quadratic', 'cubic']].values.reshape(-1, 9)
y_test = carSales.loc[81:107, 'sales'].values.reshape(-1, 1)

regressor_complex.fit(x_complex_train, y_train)

x_train = carSales.loc[0:80, 'period'].values.reshape(-1, 1)
x_test = carSales.loc[81:107, 'period'].values.reshape(-1, 1)
x_trig_train = carSales.loc[0:80, ['period', 'cos_period', 'sin_period']].values.reshape(-1, 3)
x_trig_test = carSales.loc[81:107, ['period', 'cos_period', 'sin_period']].values.reshape(-1, 3)

regressor.fit(x_train, y_train)
regressor_trig.fit(x_trig_train, y_train)

test_predictions = [prediction for sublist in regressor.predict(x_test) for prediction in sublist]
trig_test_predictions = [prediction for sublist in regressor_trig.predict(x_trig_test) for prediction in sublist]
complex_test_predictions = [prediction for sublist in regressor_complex.predict(x_complex_test) for prediction in sublist]

print(f"\nSales Prediction RMSE: {get_rmse(test_predictions, salesList[81:107])}")
print(f"Trig Prediction RMSE: {get_rmse(trig_test_predictions, salesList[81:107])}")
print(f"Complex Prediction RMSE: {get_rmse(complex_test_predictions, salesList[81:107])}")