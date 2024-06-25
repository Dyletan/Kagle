from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

melb_house_path = r'C:\projects\Kagle\excersises\melb_data.csv'
housing_data=pd.read_csv(melb_house_path)

# костыль removing rows with missing data
housing_data = housing_data.dropna(axis=0)
# target variable
y = housing_data.Price
select_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = housing_data[select_features]

melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(X, y)

predicted_home_prices = melbourne_model.predict(X)
# one of the measures of model's performance
print(mean_absolute_error(y, predicted_home_prices))

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

