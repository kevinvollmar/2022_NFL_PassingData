import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the 2022 NFL individual passing data into a Pandas dataframe
passing_data = pd.read_csv('2022_NFL_PassingTable.csv')

# Print the first 5 rows of df to check if data loaded correctly
# print(passing_data.head())

# Checking for missing values in the dataset
# print(passing_data.isnull().sum())

# Checking for outliers
# plt.boxplot(passing_data['Yds'])
# plt.show()

# Basic statistics of the table
# print(passing_data['Yds'].describe())

# Scatter plot of touchdowns v. interceptions
# plt.scatter(passing_data['TD'], passing_data['Int'])
# plt.xlabel('Touchdowns')
# plt.ylabel('Interceptions')
# plt.show()