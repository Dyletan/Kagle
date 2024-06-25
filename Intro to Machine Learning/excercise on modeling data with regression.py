import pandas as pd

iowa_file_path = r'C:\projects\Kagle\house-prices-advanced-regression-techniques\train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_names]

# print(X.head())

from sklearn.tree import DecisionTreeRegressor
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X, y)
predictions = iowa_model.predict(X)
print(predictions)

print(iowa_model.predict(X.head()))
y.head()