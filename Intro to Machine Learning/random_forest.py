from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split

melb_house_path = r'Intro to Machine Learning\melb_data.csv'
housing_data=pd.read_csv(melb_house_path)

housing_data = housing_data.dropna(axis=0)
y = housing_data.Price
select_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = housing_data[select_features]

train_X, val_X, train_y, val_y = train_test_split(X, y) 

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))