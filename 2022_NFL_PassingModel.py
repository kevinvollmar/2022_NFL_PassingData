import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Loading the 2022 NFL individual passing data into a Pandas dataframe
passing_data = pd.read_csv('2022_NFL_PassingTable.csv')

# Drop rows with NaN values in any column
passing_data = passing_data.dropna()

X = passing_data[['TD','Int','Yds']]
y = passing_data['QBR']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)

# Calculate R-squared
r2 = r2_score(y_test, predictions)

mse = mean_squared_error(y_test, predictions)
print('Mean squared error:', mse)

# Add regression line with R-squared value to the plot
sns.regplot(x=y_test, y=predictions, scatter=True, color='red',
            line_kws={'label':"R-squared = {:.2f}".format(r2)})

plt.xlabel('Actual QBR')
plt.ylabel('Predicted QBR')
plt.title('Actual v. Predicted QBR')
plt.show()