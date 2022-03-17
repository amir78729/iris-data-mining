import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import numpy as np

# https://www.geeksforgeeks.org/ml-handle-missing-data-with-simple-imputer/
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html

data = pd.read_csv('./iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
print(data)
print(data.isna())

imputer = SimpleImputer(missing_values=np.nan, strategy='constant')

imputer = imputer.fit(data)

# Imputing the data

# data = imputer.transform(data)

data = data.dropna()

print(data)

le = preprocessing.LabelEncoder()

# column_trans = ColumnTransformer([('scaler', StandardScaler(), 2], remainder='passthrough')

data['target'] = le.fit_transform(data.target.values)

print(data)

