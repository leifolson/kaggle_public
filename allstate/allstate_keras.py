import numpy as np
import pandas as pd
import xgboost as xgb
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error


train_data = pd.read_csv('/Users/clint/Development/data/allstate/train.csv')
test_data = pd.read_csv('/Users/clint/Development/data/allstate/test.csv')

# dimensions
print "Training data shape: ", train_data.shape
print "Test data shape: ", test_data.shape

cat_vars = train_data.ix[:,1:117].columns.tolist()

train_cat_var_set = set(train_data[cat_vars].values.flatten())
test_cat_var_set = set(test_data[cat_vars].values.flatten())

# encoding categorical vars
lab_enc = LabelEncoder()
lab_enc.fit(list(train_cat_var_set.union(test_cat_var_set)))

for var in cat_vars:
    train_data[var] = lab_enc.transform(train_data[var])
    test_data[var] = lab_enc.transform(test_data[var])

eta = 0.1
max_depth = 6
subsample = 0.5
colsample_bytree = 0.7
start_time = time.time()

params = {
        "objective": "reg:linear",
        "booster" : "gbtree",
        "eval_metric": "mae",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": 0,
}

num_boost_round = 1000
early_stopping_rounds = 50
test_size = 0.3

X_train, X_valid = train_test_split(train_data, test_size=test_size, random_state=0)
y_train = X_train.loss
y_valid = X_valid.loss

dtrain = xgb.DMatrix(X_train.ix[:,1:-1], y_train)
dvalid = xgb.DMatrix(X_valid.ix[:,1:-1], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

print "Training..."
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

print "Validation..."
check = gbm.predict(xgb.DMatrix(X_valid.ix[:,1:-1]), ntree_limit=gbm.best_iteration)
score = mean_absolute_error(y_valid.tolist(), check)

print "Final validation MSE: ", score

print "Predict test set..."
test_prediction = gbm.predict(xgb.DMatrix(test_data.ix[:,1:]), ntree_limit=gbm.best_iteration)


with(open('/Users/clint/Development/data/allstate/results.csv', 'w')) as f:
    f.write('id,loss')
    
    for (ident, pred) in zip(test_data['id'], test_prediction):
        item = str(ident) + ',' + str(pred) + '\n'
        f.write(item)




