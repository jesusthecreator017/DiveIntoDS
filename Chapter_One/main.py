"""
Displaying Data with Python
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

hour = pd.read_csv('hour.csv')

print(hour.head())
print("==================================================================")

"""
Calculating Summary Statistics
"""
print(f"Column 'Count' mean: {hour['count'].mean()}")
print(f"Column 'Count' median: {hour['count'].median()}")
print(f"Column 'Count' standard deviation: {hour['count'].std()}")
print(f"Column 'Registered' Minimum: {hour['registered'].min()}")
print(f"Column 'Registered' Maximum: {hour['registered'].max()}\n")

# Describe method displays all statistics above and more in a table
print(hour.describe())
print("==================================================================")

"""
Analyzing Subsets of Data
"""
# Nighttime Data
print(f"Row 3: Col: 'count' | value: {hour.loc[3, 'count']}\n")
print(f"{hour.loc[2:4, 'registered']}\n")
print(f"Average Ridership of registered users in the morning: {hour.loc[hour['hr'] < 5, 'registered'].mean()}\n")
print(f"Average Ridership during Cold Mornings: {hour.loc[(hour['hr'] < 5) & hour['temp'] < .50, 'count'].mean()}") # Cold Mornings
print(f"Average Ridership during Warm Mornings: {hour.loc[(hour['hr'] < 5) & hour['temp'] > .50, 'count'].mean()}") # Warm Mornings
print(f"Average Ridership of Warm or Humid Mornings: {hour.loc[(hour['temp'] > .50) | (hour['hum'] > .50), 'count'].mean()}\n") # Warm Morning or High Humidity

# Seasonal Data
print(f"Average Ridership per Season:\n{hour.groupby('season')['count'].mean()}\n")
print(f"Average Ridership per Season & Holiday:\n{hour.groupby(['season', 'holiday'])['count'].mean()}\n")

"""
Visualizing Data with Matplotlib
"""
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x = hour['instant'], y = hour['count'])
plt.xlabel('Hour')
plt.ylabel('Count')
plt.title("Ridership Count by Hour")
plt.show()

hourFirst48 = hour.loc[0:48, :]
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(x = hourFirst48['instant'], y = hourFirst48['count'])
plt.xlabel('Hour')
plt.ylabel('Count')
plt.title("Ridership Count by Hour : First 48 Hours")
plt.show()

# Testing different types of plots
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.scatter(x = hourFirst48['instant'], y = hourFirst48['count'], c = 'red', marker = '+')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.title("Ridership Count by Hour : First 48 Hours")
plt.show()

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(hourFirst48['instant'], hourFirst48['count'], color = 'red', label = 'casual', linestyle = '-')
ax4.plot(hourFirst48['instant'], hourFirst48['registered'], color = 'blue', label = 'registered', linestyle = '--')
ax4.legend()
plt.show()

fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.boxplot(x = 'hr', y = 'registered', data = hour)
plt.xlabel('Hour')
plt.ylabel('Count')
plt.title('Counts by Hour')
plt.show()

fig6, ax6 = plt.subplots(figsize=(10, 6))
ax6.hist(hour['count'], bins = 80)
plt.xlabel('Ridership')
plt.ylabel('Frequency')
plt.title('Histogram of Ridership')
plt.show()

theVariables = ['hr', 'temp', 'windspeed']
hourFirst100 = hour.loc[0:100, theVariables]
sns.pairplot(data = hourFirst100, corner=True)
plt.show()

"""
Exploring Correlations
"""
print(f"Correlation between Casual and Registered riders: {hour['casual'].corr(hour['registered'])}")
print(f"Correlation between Temperature and Humidity: {hour['temp'].corr(hour['hum'])}\n")

theName = ['hr', 'temp', 'windspeed']
corMatrix = hour[theName].corr()
print(f"{corMatrix}\n")

plt.figure(figsize=(10, 6))
sns.heatmap(corMatrix, annot=True, cmap='coolwarm', fmt='.3f', xticklabels=theName, yticklabels=theName)
plt.show()

df_hm = hour.pivot_table(index = 'hr', columns = 'weekday', values = 'count')
plt.figure(figsize=(20, 10))
sns.heatmap(df_hm, fmt = "d", cmap = 'binary', linewidths = .5, vmin = 0)
plt.show()