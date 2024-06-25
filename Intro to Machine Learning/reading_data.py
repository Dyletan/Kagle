import pandas as pd

iova_house_path = r'C:\Self Learning\Data Science\Kaggle\house-prices-advanced-regression-techniques\train.csv'
housing_data=pd.read_csv(iova_house_path)
print(housing_data.describe())
print(round(housing_data['LotArea'].mean(), 2))
# also can be done using a dot
print(round(housing_data.LotArea.mean(), 2))

# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = 2024-housing_data['YearBuilt'].max()