import numpy as np
import pandas as pd
import pickle
from scipy.stats import skew
from scipy.special import boxcox1p

def preprocess(unskew):
    categorical_cols = """MSZoning Street Alley LotShape
                        LandContour	Utilities	LotConfig	LandSlope	Neighborhood	Condition1
                        Condition2	BldgType	HouseStyle
                        RoofStyle	RoofMatl Exterior1st	Exterior2nd	MasVnrType
                        ExterQual	ExterCond Foundation BsmtQual BsmtCond BsmtExposure	BsmtFinType1
                        BsmtFinType2 Heating  HeatingQC	CentralAir	Electrical
                        KitchenQual Functional FireplaceQu GarageType
                        GarageFinish GarageQual GarageCond
                        PavedDrive PoolQC	Fence	MiscFeature SaleType	SaleCondition"""

    categorical_cols = categorical_cols.split()


    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")

    
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


        transforms = [boxcox1p, np.sqrt, np.log1p]
        for feature in numeric_features:
            feat_values = train_df[feature].values
            # ignore some columns
            if feature in ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "SalePrice", "Id"]:
                continue
            else:
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
                        # don't transform because the skew didn't decrease
                        continue
                    else:
                        transform = transforms[best_transform - 1]
                        if transform == boxcox1p:
                            train_df[feature] = boxcox1p(train_df[feature].values, 0.25)
                            test_df[feature] = boxcox1p(test_df[feature].values, 0.25)
                        else:
                            train_df[feature] = transform(train_df[feature].values)
                            test_df[feature] = transform(test_df[feature].values)


    # one-hot encode the categorical columns
    train_df = pd.get_dummies(train_df, columns = categorical_cols)
    test_df = pd.get_dummies(test_df, columns = categorical_cols)

    train_targets = train_df.pop("SalePrice")


    # some columns from train_df might not be present in test_df- just add them and set them to 0
    for col in train_df.columns:
        if col not in test_df.columns:
            test_df[col] = 0


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

    train_df.to_csv("./data/train_df.csv")
    test_df.to_csv("./data/test_df.csv")

if __name__ == "__main__":
    preprocess(unskew = True)