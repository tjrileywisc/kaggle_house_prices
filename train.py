
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
import pandas as pd
import pickle
from preprocess_data import preprocess

import os

# choose whether or not to unskew the data
unskew = False

if not os.path.exists("./data/train_df.pickle"):
    preprocess(unskew)
    

train_df = pd.read_pickle("./data/train_df.pickle")
train_df_ids = train_df.pop("Id")
test_df = pd.read_pickle("./data/test_df.pickle")
test_df_ids = test_df.pop("Id")
train_targets = pickle.load(open("./data/train_targets.pickle", mode = "rb"))

train_xgb = xgb.DMatrix(train_df.values, label = train_targets)



clf = RandomForestRegressor(n_estimators=100)

X = train_df.values
y = train_targets

# objective is root mean square error
mse = make_scorer(mean_squared_error)
scores = cross_val_score(clf, X, y, cv = 5, scoring = mse)
print(scores)

clf.fit(X, y)

result = clf.predict(test_df.values)
with open("results.csv", "w") as resultfile:
    resultfile.write("Id,SalePrice\n")
    for id, r in zip(test_df_ids, result):
        line = str(id) + ", " + str(r) + "\n"
        resultfile.write(line)


    