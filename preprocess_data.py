import numpy as np
import pandas as pd
import pickle
from scipy.stats import skew
from scipy.special import boxcox1p

def unskew_feature(feature, df, transform = None):
    
    # ignore some columns
    if feature in ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "SalePrice", "Id"]:
        return df, None

    if transform is None:
        transforms = [boxcox1p, np.sqrt, np.log1p]
        feat_values = df[feature].values

        if abs(skew(feat_values)) > 1:
            # value is skewed, try a transform, keep the winner
            skews = [skew(feat_values)]
            for transform in transforms:
                if transform == boxcox1p:
                    skews.append(abs(skew(boxcox1p(feat_values, 0.25))))
                else:
                    skews.append(abs(skew(transform(feat_values))))

            best_transform = np.argmin(skews)
            if best_transform == 0:
                # don't transform because the skew didn't decrease very much
                return df, None
            else:
                transform = transforms[best_transform - 1]
                if transform == boxcox1p:
                    df[feature] = boxcox1p(df[feature].values, 0.25)
                else:
                    df[feature] = transform(df[feature].values)
    else:
        if transform == boxcox1p:
            df[feature] = boxcox1p(df[feature].values, 0.25)
        else:
            df[feature] = transform(df[feature].values)

    return df, transform
    

def preprocess(unskew):
    
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")

    # filter out high price homes (long, light tail)
    train_df = train_df[train_df["SalePrice"] < 320000]


    train_targets = train_df.pop("SalePrice")
    
    categorical_cols = train_df.dtypes[train_df.dtypes == "object"].index
    numeric_features = train_df.dtypes[train_df.dtypes != "object"].index

    # populate NaNs with zeros in numerical features
    for feat in numeric_features:
        if feat in ["YearBuilt", "SalePrice", "Id"]:
            continue
        # replace NaNs here with a large value
        elif feat in ["YearRemodAdd", "GarageYrBlt"]:
            train_df[feat].fillna(1000, inplace = True)
            test_df[feat].fillna(1000, inplace = True)
        else:
            train_df[feat].fillna(0, inplace = True)
            test_df[feat].fillna(0, inplace = True)

    if unskew:
        
        

        # go through the (numerical) features and test for skewness;
        # if skewed, try box-cox transformation or several others
        # ref: http://shahramabyari.com/2015/12/21/data-preparation-for-predictive-modeling-resolving-skewness/

        #transforms = [boxcox1p, np.sqrt, np.log1p]
        for feature in numeric_features:
            train_df, transform = unskew_feature(feature, train_df, transform = None)
            test_df, _ = unskew_feature(feature, test_df, transform = transform)

    # one-hot encode the categorical columns
    train_df = pd.get_dummies(train_df, columns = categorical_cols)
    test_df = pd.get_dummies(test_df, columns = categorical_cols)

    


    # some columns from train_df might not be present in test_df- just add them and set them to 0
    for col in train_df.columns:
        if col not in test_df.columns:
            test_df[col] = 0

    for col in test_df.columns:
        if col not in train_df.columns:
            train_df[col] = 0


    # confirm that the same number of features exist in both datasets
    assert len(train_df.columns) == len(test_df.columns)

    # shift year values to read something like 'years since...'
    # YearBuilt	YearRemodAdd yrsold GarageYrBlt
    newest_YearBuilt = np.max(train_df["YearBuilt"])
    newest_YearRemodAdd = np.max(train_df["YearRemodAdd"])
    newest_GarageYrBlt = np.max(train_df["GarageYrBlt"])

    train_df["YearBuilt"] = newest_YearBuilt - train_df["YearBuilt"]
    train_df["YearRemodAdd"] = newest_YearRemodAdd - train_df["YearRemodAdd"]
    train_df["GarageYrBlt"] = newest_GarageYrBlt - train_df["GarageYrBlt"]
    test_df["YearBuilt"] = newest_YearBuilt - test_df["YearBuilt"]
    test_df["YearRemodAdd"] = newest_YearRemodAdd - test_df["YearRemodAdd"]
    test_df["GarageYrBlt"] = newest_GarageYrBlt - test_df["GarageYrBlt"]


    # make sure we don't have any NaNs
    for feat in train_df.columns:
        if pd.isnull(train_df[feat]).any():
            print(feat)

    pickle.dump(train_targets, open("./data/train_targets.pickle", "wb"))
    train_df.to_pickle("./data/train_df.pickle")
    test_df.to_pickle("./data/test_df.pickle")

if __name__ == "__main__":
    preprocess(unskew = True)