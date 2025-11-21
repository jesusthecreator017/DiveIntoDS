"""
Cleaning Erroneous Data
"""
# Import Libraries
import pandas as pd

# Load the data into memory
carSales = pd.read_csv('carsales.csv')

# Peek at the data
print(f"First Five Rows:\n{carSales.head()}")

# Reformat the column names
carSales.columns = ['month', 'sales']

# Peek at the bottom of the data (We can see the last row containes a NaN
print(f"Last Five Rows:\n{carSales.tail()}")

carSales = carSales.loc[:107, :]

print(f"Last Five Rows:\n{carSales.tail()}")

