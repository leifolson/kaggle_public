'''
Main driver program for processing data and running models
'''

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn import cross_validation

def map_cat_vars(df, encoder_map):
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = encoder_map[c].transform(df[c])


# read in the data
print 'reading data...'
data_src = '~/Development/data/house_prices/'

train_data = pd.read_csv(data_src + 'train.csv', index_col='Id')
test_data = pd.read_csv(data_src + 'test.csv', index_col='Id')

# split out data and targets
targets = train_data.SalePrice
train_data = train_data.drop('SalePrice', axis=1)

# map variables
print 'mapping vars...'
encoder_map = {}

joined = pd.concat([train_data, test_data])

# impute missing values
for c in joined.columns:
    if joined[c].dtype != 'object':
        train_data[c] = train_data[c].replace(np.nan, train_data[c].mean())
        test_data[c] = test_data[c].replace(np.nan, test_data[c].mean()) 

# create encoders
for c in joined.columns:
    if joined[c].dtype == 'object':
        le = preprocessing.LabelEncoder()
        le.fit(joined[c])
        encoder_map[c] = le

# replace categorical vars with mapped values
map_cat_vars(train_data, encoder_map)
map_cat_vars(test_data, encoder_map)
 
# create cv datasets
print 'create cv datasets...'
X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(train_data, targets, 
                                                                 test_size=0.3, random_state = 0)

# trying out a boosted tree
dtrain = xgb.DMatrix(X_train, label=y_train)
dcv = xgb.DMatrix(X_cv, y_cv)
dtest = xgb.DMatrix(test_data)

# set up params
param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'reg:linear', 'eval_metric':'rmse' }
evallist  = [(dcv,'eval'), (dtrain,'train')]

num_rounds = 1000
bst = xgb.train(param, dtrain, num_rounds, evallist, early_stopping_rounds = 10)

preds = bst.predict(dtest)

# create result set
test_data['SalePrice'] = preds
test_data.to_csv('~/Development/data/house_prices/boost_sub_1.csv', columns=['SalePrice'])





