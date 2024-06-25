import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

iowa_file_path = r'C:\projects\Kagle\house-prices-advanced-regression-techniques\train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)

rf_val_mae = mean_absolute_error(rf_model.predict(val_X), val_y)
print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
