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
import matplotlib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning,)
warnings.filterwarnings("ignore", category=UserWarning,)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None


def linear_reg(train_data, train_labels, test_data, test_labels,
                     pred_set=None,pred_set_id=None):

    reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1,
                     normalize=False)
    reg.fit(train_data, train_labels)


    pred = reg.predict(test_data)
    print("Linear Regression")
    print("RMSLE ", mean_squared_log_error(test_labels, pred))
    print("RMSE ", mean_squared_error(test_labels, pred))
    print("Variance ", explained_variance_score(test_labels, pred))
    print("R2 ", r2_score(test_labels, pred))


    if pred_set is not None:
        v_pred = reg.predict(pred_set)
        v_pred = np.exp(v_pred)
        res = pd.DataFrame({"Id": pred_set_id, "SalePrice": v_pred})
        res.to_csv("predictions.csv", index=False)

def lasso(train_data, train_labels, test_data, test_labels,
                     pred_set=None,pred_set_id=None):

    from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
    alpha = [x for x in np.arange(start=0.0, stop=1, step=0.01)]
    # Number of features to consider at every split
    fit_intercept = [True, False]
    # Maximum number of levels in tree
    normalize = [True, False]
    precompute = [True, False]
    # Minimum number of samples required to split a node
    max_iter = [x for x in np.arange(start=100, stop=1000, step=100)]
    # Minimum number of samples required at each leaf node
    tol = [x for x in np.arange(start=0.0, stop=1, step=0.01)]
    # Method of selecting samples for training each tree
    positive = [True, False]  # Create the random grid
    random_grid = {'alpha': alpha,
                   'fit_intercept': fit_intercept,
                   'normalize': normalize,
                   'precompute': precompute,
                   'max_iter': max_iter,
                   'tol': tol}

    reg = linear_model.Lasso(tol=0.04, precompute=True, normalize=True,
                             max_iter=200,fit_intercept=True, alpha=0.0,
                             random_state=42)

    #rf_random = RandomizedSearchCV(estimator=reg, param_distributions=random_grid, n_iter=1000, cv=3, verbose=2,random_state=42, n_jobs=-1)  # Fit the random search model
    #rf_random.fit(train_data, train_labels)
    #print(rf_random.best_params_)

    reg.fit(train_data, train_labels)


    pred = reg.predict(test_data)
    print("Lasso")
    print("RMSLE ", mean_squared_log_error(test_labels, pred))
    print("RMSE ", mean_squared_error(test_labels, pred))
    print("Variance ", explained_variance_score(test_labels, pred))
    print("R2 ", r2_score(test_labels, pred))

    # print("Validation Set")
    # pred = reg.predict(valid_set_data)
    # print(mean_squared_log_error(valid_set_labels, pred))
    # print(explained_variance_score(valid_set_labels, pred))

    if pred_set is not None:
        v_pred = reg.predict(pred_set)
        v_pred = np.exp(v_pred)
        res = pd.DataFrame({"Id": pred_set_id, "SalePrice": v_pred})
        res.to_csv("predictions.csv", index=False)


def forest_regressor(train_data, train_labels, test_data, test_labels,
                     pred_set=None,pred_set_id=None):
    # random forest regressor
    from sklearn.model_selection import GridSearchCV

    from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=2000, num=10)]
    max_features = ['auto', 'sqrt','log2', None]
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    min_weight_fraction_leaf = [x for x in np.arange(start=0.0, stop=0.5, step=0.01)]
    max_leaf_nodes = [int(x) for x in np.arange(start=5, stop=200, step=5)]
    max_leaf_nodes.append(None)
    min_impurity_decrease = [x for x in np.arange(start=0.0, stop=1, step=0.01)]
    bootstrap = [True, False]  # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'min_weight_fraction_leaf': min_weight_fraction_leaf,
                   'max_leaf_nodes': max_leaf_nodes,
                   'min_impurity_decrease': min_impurity_decrease,
                   }

    from sklearn.ensemble import RandomForestRegressor
    #reg = RandomForestRegressor(random_state=42)
    reg = RandomForestRegressor(n_estimators=231, min_weight_fraction_leaf=0.05,min_samples_split=2,
                                min_samples_leaf=4, min_impurity_decrease=0.0, max_leaf_nodes=185,
                                max_features='log2',max_depth=10,
                                random_state=42, n_jobs=-1)

    # rf_random = RandomizedSearchCV(estimator=reg, param_distributions=random_grid, n_iter=2000, cv=3, verbose=2,random_state=42, n_jobs=-1)  # Fit the random search model
    # rf_random.fit(train_data, train_labels)
    # print(rf_random.best_params_)

    reg.fit(train_data, train_labels)

    pred = reg.predict(test_data)
    print("Forest Regressor")
    print("RMSLE ", mean_squared_log_error(test_labels, pred))
    print("RMSE ", mean_squared_error(test_labels, pred))
    print("Variance ", explained_variance_score(test_labels, pred))
    print("R2 ", r2_score(test_labels, pred))

    #print("Validation Set")
    #pred = reg.predict(valid_set_data)
    #print(mean_squared_log_error(valid_set_labels, pred))
    #print(explained_variance_score(valid_set_labels, pred))

    if pred_set is not None:
        v_pred = reg.predict(pred_set)
        v_pred = np.exp(v_pred)
        res = pd.DataFrame({"Id":pred_set_id,"SalePrice":v_pred})
        res.to_csv("predictions.csv", index=False)

    ##########################################################################

def ridge(train_data, train_labels, test_data, test_labels,
                     pred_set=None,pred_set_id=None):
    # from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
    # alpha = [x for x in np.arange(start=0.0, stop=1, step=0.01)]
    # fit_intercept = [True, False]
    # solver = ['svd', 'cholesky', 'lsqr', 'sparse_cg']
    # max_iter = [x for x in np.arange(start=100, stop=5000, step=100)]
    # tol = [x for x in np.arange(start=0.0, stop=1, step=0.01)]
    # random_grid = {'alpha': alpha,
    #                'fit_intercept': fit_intercept,
    #                'solver': solver,
    #                'max_iter': max_iter,
    #                'tol': tol}

    reg = linear_model.Ridge(tol=0.06, solver="svd", max_iter=4500, fit_intercept=True,
                             alpha=0.46, random_state=42)

    # rf_random = RandomizedSearchCV(estimator=reg, param_distributions=random_grid, n_iter=2000, cv=3, verbose=2,random_state=42, n_jobs=-1)  # Fit the random search model
    # rf_random.fit(train_data, train_labels)
    # print(rf_random.best_params_)



    reg.fit(train_data, train_labels)

    pred = reg.predict(test_data)
    print("Ridge")
    print("RMSLE ",mean_squared_log_error(test_labels, pred))
    print("RMSE ",mean_squared_error(test_labels, pred))
    print("Variance ",explained_variance_score(test_labels, pred))
    print("R2 ", r2_score(test_labels,pred))
    #print("Validation Set")
    #pred = reg.predict(valid_set_data)
    #print(mean_squared_log_error(valid_set_labels, pred))
    #print(explained_variance_score(valid_set_labels, pred))

    if pred_set is not None:
        v_pred = reg.predict(pred_set)
        v_pred = np.exp(v_pred)
        res = pd.DataFrame({"Id":pred_set_id,"SalePrice":v_pred})
        res.to_csv("predictions.csv", index=False)

    ##########################################################################


def data_pipeline(d):
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

    scale_log_attribs = ['LotFrontage','MasVnrArea','LotArea',
                         '1stFlrSF','2ndFlrSF','BsmtFinSF1',
                         'BsmtUnfSF','GrLivArea',
                         'OpenPorchSF','TotalBsmtSF','WoodDeckSF']

    for att in scale_log_attribs:
        try:
            d[att] = d[att].apply(np.log)
            d[att] = d[att].replace(-np.inf, 0)
        except KeyError:
            print("Warning, KeyError, attrib not found in data")
            continue


    factorization_attribs = list(d.select_dtypes(include=[np.object]).columns)
    factorization_attribs = factorization_attribs + ['YearBuilt','YearRemodAdd','YrSold',
                                                     'MoSold','GarageYrBlt','MSSubClass']
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

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    train_labels = train_data['SalePrice'].apply(np.log)
    train_data = train_data.drop(columns='SalePrice')


    test_labels = test_data['SalePrice'].apply(np.log)
    test_data = test_data.drop(columns='SalePrice')

    train_data = data_pipeline(train_data)
    test_data = data_pipeline(test_data)

    pred_set = pd.read_csv('test.csv')
    pred_set_id = pred_set['Id']
    pred_set = pred_set.drop(columns=dropped_attribs)
    pred_set = data_pipeline(pred_set)

    while True:
        from sklearn import linear_model
        from regressors import stats
        ols = linear_model.LinearRegression()

        ols = ols.fit(train_data, train_labels)


        xlabels = list(train_data.columns)
        statCoef = list(stats.coef_pval(ols, train_data,train_labels))

        h = 0
        for x in range(len(statCoef)-1):
            if h < statCoef[x+1]:
                h = statCoef[x+1]

                #0.05
        if h > 0.07:
            idx = statCoef.index(h) - 1
            train_data = train_data.drop(columns=xlabels[idx])
            test_data = test_data.drop(columns=xlabels[idx])
            pred_set = pred_set.drop(columns=xlabels[idx])

        else:
            break

    stats.summary(ols, train_data, train_labels, xlabels=xlabels)

    #forest_regressor(train_data, train_labels, test_data, test_labels,pred_set,pred_set_id)

    linear_reg(train_data, train_labels, test_data, test_labels,pred_set,pred_set_id)

    lasso(train_data, train_labels, test_data, test_labels,pred_set,pred_set_id)

    ridge(train_data, train_labels, test_data, test_labels,pred_set,pred_set_id)


data = pd.read_csv('iowaHomes.csv')


from sklearn.neighbors import LocalOutlierFactor
outliers = data.loc[data['SalePrice'] > 50000]
outliers = outliers[['SalePrice']]
outliers['outlier'] = LocalOutlierFactor(n_neighbors=100, contamination=0.1).fit_predict(outliers)
outliers = outliers.loc[outliers['outlier'] == -1].index.values
data = data.drop(outliers)

outliers = data.loc[data['OverallQual'] > 3]
outliers = outliers[['OverallQual','SalePrice']]
outliers['outlier'] = LocalOutlierFactor(n_neighbors=150, contamination=0.1).fit_predict(outliers)
outliers = outliers.loc[outliers['outlier'] == -1].index.values
print(outliers)
print(len(outliers))

plt.scatter(data['OverallQual'], data['SalePrice'], alpha=0.2)
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('plots/scatterplots/scatter_' + str('OverallQual') + '.png')
plt.close()


data = data.drop(outliers)


#data = data_pipeline(data)
# for column in data:
#     plt.scatter(data[column], data['SalePrice'], alpha=0.5)
#     plt.xlabel(column)
#     plt.ylabel('SalePrice')
#     fig = matplotlib.pyplot.gcf()
#     fig.set_size_inches(18.5, 10.5)
#     plt.savefig('plots/scatterplots/scatter_' + str(column) + '.png')
#     plt.close()
# exit()

#data = data.drop(data[(data['SalePrice'] > 550000)].index)

# data = data.drop(data[(data['OverallQual'] < 5.0) & (data['SalePrice'] > 200000)].index)
# data = data.drop(data[(data['OverallQual'] == 8.0) & (data['SalePrice'] > 470000)].index)
# data = data.drop(data[(data['OverallQual'] == 9.0) & (data['SalePrice'] > 430000)].index)
# data = data.drop(data[(data['OverallQual'] == 7.0) & (data['SalePrice'] > 360000)].index)
# data = data.drop(data[(data['OverallQual'] == 7.0) & (data['SalePrice'] < 90000)].index)


data = data.drop(data[(data['YearBuilt'] < 1920) & (data['SalePrice'] > 250000)].index)
data = data.drop(data[(data['YearBuilt'] < 1960) & (data['SalePrice'] > 300000)].index)


data = data.drop(data[data['LotArea'] > 35000].index)



data = data.drop(data[(data['OverallCond'] == 2.0) & (data['SalePrice'] > 330000)].index)

data = data.drop(data[(data['MSZoning'] == "RM") & (data['SalePrice'] > 300000)].index)


#data = data.drop([341,348])
#
# plt.scatter(data['BsmtFinSF1'], data['SalePrice'], alpha=0.5)
# plt.xlabel('BsmtFinSF1')
# plt.ylabel('SalePrice')
#plt.show()
#exit()


#


#print(data['SalePrice'])
#exit()

dropped_attribs = ["Id", "Alley",
                    "Street",'PoolQC','3SsnPorch',"2ndFlrSF",
                   'EnclosedPorch','Utilities','RoofStyle',
                   'RoofMatl',"PoolArea"
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
#data, valid_set = train_test_split(data, test_size=0.2, random_state=42)

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
