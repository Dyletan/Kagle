import pandas as pd

melb_house_path = r'C:\Self Learning\Data Science\Kaggle\Intro to Machine Learning\melb_data.csv'
housing_data=pd.read_csv(melb_house_path)

print(housing_data.columns)

# костыль removing rows with missing data
housing_data = housing_data.dropna(axis=0)
# target variable
y = housing_data.Price

select_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = housing_data[select_features]

# print('\n', X.describe())
# print(X.head())

from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))