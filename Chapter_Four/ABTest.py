# Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


# Load Data
desktop = pd.read_csv('desktop.csv')
laptop = pd.read_csv('laptop.csv')

# Quick T-Tests on columns
print(f"P-Value from T-Test for Spending: {stats.ttest_ind(desktop['spending'], laptop['spending'])[1]}")
print(f"P-Value from T-Test for Age: {stats.ttest_ind(desktop['age'], laptop['age'])[1]}")
print(f"P-Value from T-Test for Visits: {stats.ttest_ind(desktop['visits'], laptop['visits'])[1]}")

deskMedianAge = desktop['age'].median()
groupA = desktop.loc[desktop['age'] <= deskMedianAge]
groupB = desktop.loc[desktop['age'] > deskMedianAge]

emailResults = pd.read_csv('emailresults1.csv')
print(f"\nEmail Results Peek:\n{emailResults.head()}")

groupAWithRev = groupA.merge(emailResults, on='userid')
groupBWithRev = groupB.merge(emailResults, on='userid')

print(f"\nP-Value from T-Test from Email Data: {stats.ttest_ind(groupAWithRev['revenue'], groupBWithRev['revenue'])[1]}")
print(f"Mean Difference between Groups A-B: {np.mean(groupBWithRev['revenue'] - np.mean(groupAWithRev['revenue']))}")