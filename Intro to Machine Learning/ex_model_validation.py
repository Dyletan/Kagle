import pandas as pd
from sklearn.tree import DecisionTreeRegressor

iowa_file_path = r"C:\projects\Kagle\house-prices-advanced-regression-techniques\train.csv"
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

iowa_model = DecisionTreeRegressor()
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)

# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)

# print the top few validation predictions
print(val_predictions[:7])
# print the top few actual prices from validation data
print(val_y.to_list()[:7])

from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_predictions, val_y)
print(val_mae)