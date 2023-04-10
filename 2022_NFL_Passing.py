import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Loading the 2022 NFL individual passing data into a Pandas dataframe
passing_data = pd.read_csv('2022_NFL_PassingTable.csv')

#Drop rows with NaN values in any column
passing_data = passing_data.dropna()

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

# Seaborn data analysis looking to see if there is correlation between yards per attempt and sack %. Along with colorizing by age
# filtered_subset = passing_data.loc[passing_data['Att'] >= 100]
# sns.scatterplot(data=filtered_subset, x='Y/A', y='Sk%', hue='Age')
# plt.show()

#X = passing_data.drop(columns=['QBR'])
X = passing_data[['Cmp','Att', 'TD','Int','Yds','Sk']]
y = passing_data['QBR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)

mse = mean_squared_error(passing_data['QBR'], predictions)
print("Mean squared error:", mse)