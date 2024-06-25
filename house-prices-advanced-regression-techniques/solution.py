import pandas as pd

iova_house_path = r'C:\projects\Kagle\house-prices-advanced-regression-techniques\train.csv'
housing_data=pd.read_csv(iova_house_path)
print(housing_data.describe())
round(housing_data['LotArea'].mean(), 6)