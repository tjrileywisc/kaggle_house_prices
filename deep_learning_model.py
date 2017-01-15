
import pandas as pd
import numpy as np
import pickle

from keras import backend as K
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import StandardScaler

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


train_df = pd.read_pickle("./data/train_df.pickle")
train_df_ids = train_df.pop("Id")
test_df = pd.read_pickle("./data/test_df.pickle")
test_df_ids = test_df.pop("Id")

y = pickle.load(open("./data/train_targets.pickle", mode = "rb"))
max_y = np.max(y)
y /= max_y

X = train_df.values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# it's not a lot of data, so make a small network
input = Input(shape = (X.shape[1],))
x = Dense(128, activation = 'relu')(input)
preds = Dense(1, activation = 'sigmoid')(x)

model = Model(input = input, output = preds)

model.compile(optimizer = 'adam', loss = root_mean_squared_error)

model.fit(X, y, nb_epoch=100)

X_test = scaler.transform(test_df.values)
result = model.predict(X_test)
result *= max_y

with open("results.csv", "w") as resultfile:
    resultfile.write("Id,SalePrice\n")
    for id, r in zip(test_df_ids, result):
        line = str(id) + ", " + str(r[0]) + "\n"
        resultfile.write(line)