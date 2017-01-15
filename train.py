
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from preprocess_data import preprocess

import os


# objective is root mean square error
def rmse_loss(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# choose whether or not to unskew the data
unskew = True

if not os.path.exists("./data/train_df.pickle"):
    preprocess(unskew)
    

train_df = pd.read_pickle("./data/train_df.pickle")
train_df_ids = train_df.pop("Id")
test_df = pd.read_pickle("./data/test_df.pickle")
test_df_ids = test_df.pop("Id")
train_targets = pickle.load(open("./data/train_targets.pickle", mode = "rb"))

clf = RandomForestRegressor(n_estimators=100)

X = train_df.values
y = train_targets

# objective is root mean square error
rmse = make_scorer(rmse_loss, greater_is_better=False)

scores = cross_val_score(clf, X, y, cv = 5, scoring = rmse)
print(scores)

clf.fit(X, y)

# get the feature importances
importances = clf.feature_importances_

X = X[:, importances > 1e-4]
# fit again with the filtered features
clf.fit(X, y)

test_features = test_df.values[:, importances > 1e-4]


result = clf.predict(test_features)
with open("results.csv", "w") as resultfile:
    resultfile.write("Id,SalePrice\n")
    for id, r in zip(test_df_ids, result):
        line = str(id) + ", " + str(r) + "\n"
        resultfile.write(line)


    