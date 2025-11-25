# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the data
mlb = pd.read_csv("mlb.csv")

# Print data and it's stats
print(f"First Five Entries:\n{mlb.head()}\n")
print(f"Data Shape: {mlb.shape}\n")
print(f"Data stats:\n{mlb.describe()}\n")

# Plot the data
plt1, ax1 = plt.subplots()
ax1.boxplot(mlb['height'])
ax1.set_ylabel("Height (Inches)")
plt.title("MLB Player Heights")
plt.xticks([1], ['Full Population'])
plt.show()

# Create Samples
sample1 = mlb.sample(n=30, random_state=8675309)
sample2 = mlb.sample(n=30, random_state=1729)
sample3 = [71, 72, 73, 74, 74, 76, 75, 75, 75, 76, 75, 77, 76, 75, 77, 76, 75, 76, 76, 75, 75, 81, 77, 75, 77, 75, 77, 77, 75, 75]

# Plot samples
fig1, ax1 = plt.subplots()
ax1.boxplot([mlb['height'], sample1['height'], sample2['height'], np.array(sample3)])
ax1.set_ylabel("Height (Inches)")
plt.title("MLB Player Heights")
plt.xticks([1, 2, 3, 4], ['Full Population', 'Sample 1', 'Sample 2', 'Sample 3'])
plt.show()

# Print means
print(f"Sample 1 Mean: {np.mean(sample1['height'])}")
print(f"Sample 2 Mean: {np.mean(sample2['height'])}")
print(f"Sample 3 Mean: {np.mean(sample3)}")

"""
Differences Between Sample Data
"""

allDifferences = []

for i in range(1000):
    newSample1 = mlb.sample(n=30, random_state= i * 2)
    newSample2 = mlb.sample(n=30, random_state=i * 2 + 1)
    allDifferences.append(newSample1['height'].mean() - newSample2['height'].mean())

print(allDifferences[0:10])

# Plot all 1000 differences
sns.set_style()
sns.histplot(allDifferences).set_title("Differences Between Sample Means")
plt.xlabel("Differences Between Means (Inches)")
plt.ylabel("Relative Frequency")
plt.show()

# Check how many differences from sample3 v.s sample1 are grater than 1.6
largedDifferences = [diff for diff in allDifferences if abs(diff) >= 1.6]
print(f"Amount of differences larger than 1.6: {len(largedDifferences)}")

smallDifferences = [diff for diff in allDifferences if abs(diff) >= 0.6]
print(f"Amount of differences smaller than 0.6: {len(smallDifferences)}")

"""
Performing Hypothesis Testing
"""
# T-test implementation
print(stats.ttest_ind(sample1['height'], sample2['height']))
print(stats.ttest_ind(sample1['height'], sample3))

# Mann-Whitney U Test Implementation
print(stats.mannwhitneyu(sample1['height'], sample2['height']))