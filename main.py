import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None

data = pd.read_csv('iowaHomes.csv')


dropped_attribs = ["Id", "Alley", "PoolArea",
                   "MoSold", "3SsnPorch", "BsmtFinSF2",
                   "BsmtHalfBath", "MiscVal", "LowQualFinSF",
                   "YrSold", "OverallCond", "ScreenPorch"]

data = data.drop(columns=dropped_attribs)
data, valid_set = train_test_split(data, test_size=0.2, random_state=42)




# data = factorize_data(data)
# data = data.drop(columns=['SalePrice'])
# Q1 = data.quantile(0.25)
# Q3 = data.quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# print(data.shape)
# data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
# print(data.shape)
# exit()
#
# print(data['EnclosedPorch'].value_counts())
# print(data.describe())
# print(data.shape)
# print(math.sqrt(data.shape[0]))

# import matplotlib.pyplot as plt
# for column in data:
#     plt.figure()
#     data.boxplot([column])
#     plt.savefig('plots/boxplot_' + str(column) + '.png')


#
# for x in range(data.shape[1]):
#     col = data.iloc[:,x]
#     if len(col.unique()) < 5:
#         print(col.value_counts())
# print(data)

class DropLowUniqueCols(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for x in range(data.shape[1]):
            col = X.iloc[:, x]
            if len(col.unique()) < self.threshold:
                print(col.value_counts())

# data = clean_data(data)
#
# valid_set = clean_data(valid_set)
#
# pred_set_id = pred_set['Id']
# pred_set = clean_data(pred_set)

#trying out different data scaling functions
# distributions = [
#     ('Unscaled data', "NA"),
#     ('Data after min-max scaling',
#         MinMaxScaler().fit_transform),
#     ('Data after max-abs scaling',
#         MaxAbsScaler().fit_transform),
#     ('Data after robust scaling',
#         RobustScaler(quantile_range=(25, 75)).fit_transform),
#     ('Data after power transformation (Yeo-Johnson)',
#      PowerTransformer(method='yeo-johnson').fit_transform),
#     ('Data after power transformation (Box-Cox)',
#      PowerTransformer().fit_transform),
#     ('Data after quantile transformation (uniform pdf)',
#         QuantileTransformer(output_distribution='uniform')
#         .fit_transform),
#     ('Data after sample-wise L2 normalizing',
#         Normalizer().fit_transform),
#     ('Data after standardization',
#      StandardScaler().fit_transform)
# ]

# distributions = [
#     (PowerTransformer(method='yeo-johnson').fit_transform),
# ]

def linear_reg(train_data, train_labels, test_data, test_labels,
                     valid_set_data, valid_set_labels,
                     pred_set=None,pred_set_id=None):

    reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1,
                     normalize=False)
    reg.fit(train_data, train_labels)


    pred = reg.predict(test_data)
    print("Test set")
    print(mean_squared_log_error(test_labels, pred))
    print(explained_variance_score(test_labels, pred))

    print("Validation Set")
    pred = reg.predict(valid_set_data)
    print(mean_squared_log_error(valid_set_labels, pred))
    print(explained_variance_score(valid_set_labels, pred))

    if pred_set is not None:
        v_pred = reg.predict(pred_set)
        res = pd.DataFrame({"Id":pred_set_id,"SalePrice":v_pred})
        res.to_csv("predictions.csv")

def forest_regressor(train_data, train_labels, test_data, test_labels,
                     valid_set_data, valid_set_labels,
                     pred_set=None,pred_set_id=None):
    # random forest regressor
    from sklearn.model_selection import GridSearchCV

    # param_grid = [
    #     {'n_estimators': [50, 100, 150, 200], 'max_features': [4, 8, 32, 64], 'n_jobs': [-1]},
    #     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    # ]

    from sklearn.ensemble import RandomForestRegressor
    reg = RandomForestRegressor(verbose=True, n_estimators=150, max_features=32, random_state=42, n_jobs=-1)

    # grid_search = GridSearchCV(reg, param_grid, cv=5,scoring='r2')
    # print(grid_search.best_estimator_)

    reg.fit(train_data, train_labels)


    pred = reg.predict(test_data)
    print("Test set")
    print(mean_squared_log_error(test_labels, pred))
    print(explained_variance_score(test_labels, pred))

    print("Validation Set")
    pred = reg.predict(valid_set_data)
    print(mean_squared_log_error(valid_set_labels, pred))
    print(explained_variance_score(valid_set_labels, pred))

    if pred_set is not None:
        v_pred = reg.predict(pred_set)
        res = pd.DataFrame({"Id":pred_set_id,"SalePrice":v_pred})
        res.to_csv("predictions.csv")

    ##########################################################################

class PipelineAwareLabelEncoder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return OneHotEncoder().fit_transform(X).values.reshape(-1,1)

def data_pipeline(d):
    custom_cat_attribs = ['FireplaceQu', 'BsmtQual', 'BsmtCond',
                          'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                          'GarageType', 'GarageQual', 'GarageFinish',
                          'PoolQC', 'Fence', 'MiscFeature', 'GarageCond']

    naImputer = SimpleImputer(strategy="constant", fill_value="NA")
    for att in custom_cat_attribs:
        d[att] = naImputer.fit_transform(d[att].values.reshape(-1, 1))

    cat_attribs = list(d.select_dtypes(include=[np.object]).columns)
    freqImputer = SimpleImputer(strategy="most_frequent")
    for att in cat_attribs:
        d[att] = freqImputer.fit_transform(d[att].values.reshape(-1, 1))

    num_attribs = list(d.select_dtypes(include=[np.number]).columns)
    numImputer = SimpleImputer(strategy="median")
    for att in num_attribs:
        d[att] = numImputer.fit_transform(d[att].values.reshape(-1, 1))

    factorization_attribs = ['MSSubClass', 'MSZoning', 'Street',
                             'LotShape', 'LandContour', 'Utilities',
                             'LotConfig', 'LandSlope', 'Neighborhood',
                             'Condition1', 'Condition2', 'BldgType',
                             'HouseStyle', 'RoofStyle', 'RoofMatl',
                             'Exterior1st', 'Exterior2nd', 'MasVnrType',
                             'ExterQual', 'ExterCond', 'Foundation',
                             'BsmtQual', 'BsmtCond', 'BsmtExposure',
                             'BsmtFinType1', 'BsmtFinType2', 'Heating',
                             'HeatingQC', 'CentralAir', 'Electrical',
                             'KitchenQual', 'Functional', 'FireplaceQu',
                             'GarageType', 'GarageFinish', 'GarageQual',
                             'GarageCond', 'PavedDrive', 'PoolQC',
                             'Fence', 'MiscFeature', 'SaleType',
                             'SaleCondition'
                             ]
    #enc = OneHotEncoder()
    enc = OrdinalEncoder()
    for att in factorization_attribs:
        temp = d[att].values.reshape(-1, 1)
        xf = enc.fit(temp)
        d[att] = xf.transform(temp)

    return d

def main():

    from sklearn.model_selection import KFold

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)



    train_labels = train_data['SalePrice']
    train_data = train_data.drop(columns='SalePrice')

    test_labels = test_data['SalePrice']
    test_data = test_data.drop(columns='SalePrice')

    valid_set_labels = valid_set['SalePrice']
    valid_set_data = valid_set.drop(columns='SalePrice')


    train_data = data_pipeline(train_data)
    test_data = data_pipeline(test_data)
    valid_set_data = data_pipeline(valid_set_data)

    pred_set = pd.read_csv('test.csv')
    pred_set_id = pred_set['Id']
    pred_set = pred_set.drop(columns=dropped_attribs)
    pred_set = data_pipeline(pred_set)

    forest_regressor(train_data, train_labels, test_data, test_labels,
                     #valid_set_data,valid_set_labels,pred_set,pred_set_id)

    #linear_reg(train_data, train_labels, test_data, test_labels,
                     #valid_set_data,valid_set_labels,pred_set,pred_set_id)
    exit()

    #
    # #clean_data = prep_data(data)
    # clean_pred_set = prep_data(pred_set)
    # clean_valid_set = prep_data(valid_set)
    # # kf = KFold(n_splits=10, random_state=42)
    # # for train, test in kf.split(cleanData):
    # #
    # #     train_data = cleanData.drop(columns='SalePrice').iloc[train]
    # #     train_labels = cleanData['SalePrice'].iloc[train]
    # #
    # #     test_data = cleanData.drop(columns='SalePrice').iloc[test]
    # #     test_labels = cleanData['SalePrice'].iloc[test]
    #
    # # x = cleanData.drop(columns='SalePrice')
    # # y = cleanData['SalePrice']
    #
    #

main()













