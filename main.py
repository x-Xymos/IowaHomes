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

import warnings
warnings.filterwarnings("ignore", category=UserWarning,)
warnings.filterwarnings("ignore", category=RuntimeWarning,)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None

# class DropLowUniqueCols(BaseEstimator, TransformerMixin):
#     def __init__(self, threshold):
#         self.threshold = threshold
#         pass
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         for x in range(data.shape[1]):
#             col = X.iloc[:, x]
#             if len(col.unique()) < self.threshold:
#                 print(col.value_counts())


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

    from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]  # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    from sklearn.ensemble import RandomForestRegressor

    reg = RandomForestRegressor(n_estimators=400, min_samples_split=2,
                                min_samples_leaf=1, max_features='sqrt',max_depth=None,
                                bootstrap=False,
                                random_state=42, n_jobs=-1)
    #rf_random = RandomizedSearchCV(estimator=reg, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,random_state=42, n_jobs=-1)  # Fit the random search model
    #rf_random.fit(train_data, train_labels)
    #print(rf_random.best_params_)

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
        v_pred = np.exp(v_pred)
        res = pd.DataFrame({"Id":pred_set_id,"SalePrice":v_pred})
        res.to_csv("predictions.csv", index=False)

    ##########################################################################


def data_pipeline(d, scale_func=None):
    custom_cat_attribs = ['FireplaceQu', "PoolQC", 'BsmtQual', 'BsmtCond',
                          'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                          'GarageType', 'GarageQual', 'GarageFinish',
                          'Fence', 'MiscFeature', 'GarageCond']
    naImputer = SimpleImputer(strategy="constant", fill_value="NA")
    for att in custom_cat_attribs:
        try:
            d[att] = naImputer.fit_transform(d[att].values.reshape(-1, 1))
        except KeyError:
            print("Warning, KeyError, attrib not found in data")
        continue

    cat_attribs = list(d.select_dtypes(include=[np.object]).columns)
    freqImputer = SimpleImputer(strategy="most_frequent")
    for att in cat_attribs:
        try:
            d[att] = freqImputer.fit_transform(d[att].values.reshape(-1, 1))
        except KeyError:
            print("Warning, KeyError, attrib not found in data")
        continue

    num_attribs = list(d.select_dtypes(include=[np.number]).columns)
    numImputer = SimpleImputer(strategy="median")
    for att in num_attribs:
        try:
            d[att] = numImputer.fit_transform(d[att].values.reshape(-1, 1))
        except KeyError:
            print("Warning, KeyError, attrib not found in data")
        continue

    scale_log_attribs = ['1stFlrSF','2ndFlrSF','BsmtFinSF1',
                         'BsmtUnfSF','GrLivArea','LotArea',
                         'OpenPorchSF','TotalBsmtSF','WoodDeckSF']
    #print(d['2ndFlrSF'])
    for att in scale_log_attribs:
        col = d[att]
        print(col)
        idx = col.to_numpy().nonzero()[0]
        print(col.iloc[idx])
        break
        try:
            print('')
            #d[att][[d[att].to_numpy().nonzero()[0]]] = d[att][[d[att].to_numpy().nonzero()[0]]].apply(np.log)
        except KeyError:
            print("Warning, KeyError, attrib not found in data")
            continue
    #print(d['2ndFlrSF'])
    exit()

    factorization_attribs = list(d.select_dtypes(include=[np.object]).columns)
    factorization_attribs = factorization_attribs + ['YearBuilt','YearRemodAdd','YrSold',
                                                     'MoSold','GarageYrBlt']
    # #enc = LabelEncoder()
    # #enc = MultiLabelBinarizer()
    enc = OrdinalEncoder()
    for att in factorization_attribs:
        try:
            d[att] = enc.fit_transform(d[att].values.reshape(-1, 1))
        except KeyError:
            print("Warning, KeyError, attrib not found in data")
        continue
    return d


def main():


    from sklearn.model_selection import KFold
    #for dist in distributions:
        #print(dist[0])
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)



    train_labels = train_data['SalePrice'].apply(np.log)
    train_data = train_data.drop(columns='SalePrice')

    test_labels = test_data['SalePrice'].apply(np.log)
    test_data = test_data.drop(columns='SalePrice')

    valid_set_labels = valid_set['SalePrice'].apply(np.log)
    valid_set_data = valid_set.drop(columns='SalePrice')

    train_data = data_pipeline(train_data)
    test_data = data_pipeline(test_data)
    valid_set_data = data_pipeline(valid_set_data)

    pred_set = pd.read_csv('test.csv')
    pred_set_id = pred_set['Id']
    pred_set = pred_set.drop(columns=dropped_attribs)
    pred_set = data_pipeline(pred_set)

    from sklearn import linear_model
    #from regressors import stats
    #ols = linear_model.LinearRegression()
    #ols.fit(train_data, train_labels)

    #print(stats.summary(ols, train_data,train_labels, xlabels=list(train_data.columns)))


    forest_regressor(train_data, train_labels, test_data, test_labels,valid_set_data,valid_set_labels,pred_set,pred_set_id)

    #linear_reg(train_data, train_labels, test_data, test_labels,valid_set_data,valid_set_labels,pred_set,pred_set_id)


data = pd.read_csv('iowaHomes.csv')

# data['SalePrice'] = data['SalePrice'].apply(np.log)
# data['1stFlrSF'] = data['1stFlrSF'].apply(np.log)
# data['BsmtFinSF1'] = data['BsmtFinSF1'].apply(np.log)
#hist_BsmtFinSF1


# import seaborn as sns
# import matplotlib
# sns.distplot(data['BsmtFinSF1'], color='blue', axlabel='BsmtFinSF1')
# fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(18.5, 10.5)
# plt.show()
# plt.close()
# exit()

# for column in data:
#     try:
#         sns.distplot(data[column], color='blue', axlabel=column)
#         fig = matplotlib.pyplot.gcf()
#         fig.set_size_inches(18.5, 10.5)
#         plt.savefig('plots/histograms/hist_' + str(column) + '.png',dpi=100)
#         plt.close()
#     except:
#         pass
#
# exit()
# for column in data:
#     plt.scatter(data['SalePrice'], data[column], alpha=0.5)
#
#     plt.xlabel('SalePrice')
#     plt.ylabel(column)
#
#
#     plt.savefig('plots/scatterplots/scatter_' + str(column) + '.png')
#     plt.close()

# for column in data:
#     try:
#         plt.boxplot(data[column], labels=[column])
#         plt.savefig('plots/boxplots/boxplot_' + str(column) + '.png')
#         plt.close()
#     except:
#         pass


dropped_attribs = ["Id", "Alley",
                    "Street",
                   ]


data = data.drop(columns=dropped_attribs)


#data = data.drop(columns=["GarageCars",'YearBuilt','1stFlrSF'])
# import seaborn as sns
# plt.figure(figsize=(12,10))
# cor = data.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()
# #Correlation with output variable
# cor_target = abs(cor["SalePrice"])#Selecting highly correlated features
# relevant_features = cor_target[cor_target>0.5]
# print(relevant_features.sort_values(ascending=False))
#
# #print(data[["OverallQual","1stFlrSF"]].corr())
# #print(data[["OverallQual","TotRmsAbvGrd"]].corr())
# #print(data[["1stFlrSF","TotRmsAbvGrd"]].corr())
#

#OverallQual #1stFlrSF SalePrice


#exit()
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

#
# for x in range(data.shape[1]):
#     col = data.iloc[:, x]
#     if len(col.unique()) < 15:
#         print(col.value_counts())
# print(data[:19])
#
# exit()

main()

# distributions = [
#     ('Unscaled data', None),
#     ('Data after min-max scaling',
#      MinMaxScaler().fit_transform),
#     ('Data after max-abs scaling',
#      MaxAbsScaler().fit_transform),
#     ('Data after robust scaling',
#      RobustScaler(quantile_range=(25, 75)).fit_transform),
#     ('Data after power transformation (Yeo-Johnson)',
#      PowerTransformer(method='yeo-johnson').fit_transform),
#     ('Data after power transformation (Box-Cox)',
#      PowerTransformer().fit_transform),
#     ('Data after quantile transformation (uniform pdf)',
#      QuantileTransformer(output_distribution='uniform')
#      .fit_transform),
#     ('Data after sample-wise L2 normalizing',
#      Normalizer().fit_transform),
#     ('Data after standardization',
#      StandardScaler().fit_transform)
# ]
