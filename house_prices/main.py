'''
Main driver program for processing data and running models
'''

import pandas as pd
from sklearn import preprocessing


def map_cat_vars(df, encoder_map):
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = encoder_map[c].transform(df[c])


# read in the data
print 'reading data...'
data_src = '~/Development/data/house_prices/'

train_data = pd.read_csv(data_src + 'train.csv')
test_data = pd.read_csv(data_src + 'test.csv')


# map variables
print 'mapping vars...'
encoder_map = {}

# create encoders
joined = pd.concat([train_data.drop('SalePrice',axis=1), test_data])
for c in joined.columns:
    if joined[c].dtype == 'object':
        le = preprocessing.LabelEncoder()
        le.fit(joined[c])
        encoder_map[c] = le


# replace categorical vars with mapped values
map_cat_vars(train_data, encoder_map)
map_cat_vars(test_data, encoder_map)
