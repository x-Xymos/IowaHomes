import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
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
    try:
        print("RMSLE ", mean_squared_log_error(test_labels, pred))
    except:
        pass
    print("RMSE ", mean_squared_error(test_labels, pred))
    print("Variance ", explained_variance_score(test_labels, pred))

    v_pred = []
    if pred_set is not None:
        v_pred = reg.predict(pred_set)
        v_pred = np.exp(v_pred)
        res = pd.DataFrame({"Id": pred_set_id, "SalePrice": v_pred})
        res.to_csv("predictionsLinearReg.csv", index=False)

    return pred, v_pred


def lasso(train_data, train_labels, test_data, test_labels,
                     pred_set=None,pred_set_id=None):

    from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
    cv = [x for x in np.arange(start=2, stop=15, step=1)]
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
    random_grid = {'cv': cv,
                   'fit_intercept': fit_intercept,
                   'normalize': normalize,
                   'precompute': precompute,
                   'max_iter': max_iter,
                   'tol': tol}
    #reg = linear_model.LassoCV()
    reg = linear_model.LassoCV(tol=0.00, precompute=True, normalize=True,
                             max_iter=300,fit_intercept=True, cv=13,
                             random_state=42)
    # rf_random = RandomizedSearchCV(estimator=reg, param_distributions=random_grid, n_iter=1000, cv=3, verbose=2,random_state=42, n_jobs=-1)  # Fit the random search model
    # rf_random.fit(train_data, train_labels)
    # print(rf_random.best_params_)

    reg.fit(train_data, train_labels)


    pred = reg.predict(test_data)
    print("Lasso")
    print("RMSLE ", mean_squared_log_error(test_labels, pred))
    print("RMSE ", mean_squared_error(test_labels, pred))
    print("Variance ", explained_variance_score(test_labels, pred))

    v_pred = []
    if pred_set is not None:
        v_pred = reg.predict(pred_set)
        v_pred = np.exp(v_pred)
        res = pd.DataFrame({"Id": pred_set_id, "SalePrice": v_pred})
        res.to_csv("predictionsLasso.csv", index=False)

    return pred, v_pred


def bayesian_ridge(train_data, train_labels, test_data, test_labels,
                     pred_set=None,pred_set_id=None):

    from sklearn.model_selection import RandomizedSearchCV

    random_grid = {'n_iter': [x for x in np.arange(start=100, stop=1000, step=5)],
                   'alpha_1': [x for x in np.arange(start=5.0, stop=50, step=0.1)],
                   'alpha_2': [x for x in np.arange(start=0.0, stop=1, step=0.0001)],
                   'lambda_1':[x for x in np.arange(start=50.0, stop=300, step=1)],
                   'lambda_2':[x for x in np.arange(start=50.0, stop=300, step=1)],
                   'fit_intercept': [True, False],
                   'normalize': [True, False],
                   'tol': [x for x in np.arange(start=5.0, stop=50, step=0.1)]}
    #reg = linear_model.BayesianRidge()

    reg = linear_model.BayesianRidge(tol=18.79999999999995, normalize=True,
                                     n_iter=245, lambda_2=191.0, lambda_1=240.0,
                                     fit_intercept=True, alpha_2=0.026000000000000002,
                                     alpha_1=13.39999999999997)

    # rf_random = RandomizedSearchCV(estimator=reg, param_distributions=random_grid, n_iter=3000, cv=3, verbose=2,random_state=42, n_jobs=-1)  # Fit the random search model
    # rf_random.fit(train_data, train_labels)
    # print(rf_random.best_params_)



    reg.fit(train_data, train_labels)

    pred = reg.predict(test_data)
    print("Bayesian Ridge")
    print("RMSLE ",mean_squared_log_error(test_labels, pred))
    print("RMSE ",mean_squared_error(test_labels, pred))
    print("Variance ",explained_variance_score(test_labels, pred))

    v_pred = []
    if pred_set is not None:
        v_pred = reg.predict(pred_set)
        v_pred = np.exp(v_pred)
        res = pd.DataFrame({"Id":pred_set_id,"SalePrice":v_pred})
        res.to_csv("predictionsBayesianRidge.csv", index=False)

    return pred, v_pred
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
    #reg = linear_model.Ridge()
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

    v_pred = []
    if pred_set is not None:
        v_pred = reg.predict(pred_set)
        v_pred = np.exp(v_pred)
        res = pd.DataFrame({"Id":pred_set_id,"SalePrice":v_pred})
        res.to_csv("predictionsRidge.csv", index=False)

    return pred, v_pred
    ##########################################################################

def gboost(train_data, train_labels, test_data, test_labels,
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
    from sklearn.ensemble import GradientBoostingRegressor
    reg = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)



    # rf_random = RandomizedSearchCV(estimator=reg, param_distributions=random_grid, n_iter=2000, cv=3, verbose=2,random_state=42, n_jobs=-1)  # Fit the random search model
    # rf_random.fit(train_data, train_labels)
    # print(rf_random.best_params_)
    reg.fit(train_data, train_labels)

    pred = reg.predict(test_data)
    print("GBOOST")
    print("RMSLE ", mean_squared_log_error(test_labels, pred))
    print("RMSE ", mean_squared_error(test_labels, pred))
    print("Variance ", explained_variance_score(test_labels, pred))

    v_pred = []
    if pred_set is not None:
        v_pred = reg.predict(pred_set)
        v_pred = np.exp(v_pred)
        res = pd.DataFrame({"Id": pred_set_id, "SalePrice": v_pred})
        res.to_csv("predictionsGBOOST.csv", index=False)

    return pred, v_pred
    ##########################################################################



def feature_engineering(d):
    d['TotalSF'] = d['TotalBsmtSF'] + d['1stFlrSF'] + d['2ndFlrSF']
    d = d.drop(columns=['1stFlrSF', '2ndFlrSF'])

    #d['Bathrooms'] = d['BsmtFullBath'] + d['BsmtHalfBath'] + d['FullBath'] + d['HalfBath']
    #d = d.drop(columns=['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'])

    d['PorchSF'] = d['OpenPorchSF'] + d['EnclosedPorch'] + d['3SsnPorch'] + d['ScreenPorch']
    d = d.drop(columns=['OpenPorchSF', 'EnclosedPorch','3SsnPorch','ScreenPorch'])

    return d


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

    # scale_log_attribs = ['LotFrontage','MasVnrArea','LotArea',
    #                      '1stFlrSF','2ndFlrSF','BsmtFinSF1','BsmtFinSF2',
    #                      'BsmtUnfSF','GrLivArea',
    #                      'OpenPorchSF','TotalBsmtSF']
    #
    # for att in scale_log_attribs:
    #     try:
    #         d[att] = d[att].apply(np.log)
    #         d[att] = d[att].replace(-np.inf, 0)
    #     except KeyError:
    #         print("Warning, KeyError, attrib not found in data")
    #         continue


    factorization_attribs = list(d.select_dtypes(include=[np.object]).columns)
    factorization_attribs = factorization_attribs + ['YearBuilt','YearRemodAdd',
                                                     'GarageYrBlt','MSSubClass']
    enc = OrdinalEncoder()
    for att in factorization_attribs:
        try:
            d[att] = enc.fit_transform(d[att].values.reshape(-1, 1))
        except KeyError:
            print("Warning, KeyError, attrib not found in data")
        continue

    return d


def main():



    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    train_labels = train_data['SalePrice'].apply(np.log)
    train_data = train_data.drop(columns='SalePrice')


    test_labels = test_data['SalePrice'].apply(np.log)
    test_data = test_data.drop(columns='SalePrice')

    train_data = data_pipeline(train_data)
    train_data = feature_engineering(train_data)

    # for column in train_data:
    #     plt.scatter(train_data[column], train_labels,  alpha=0.5)
    #
    #     plt.xlabel(column)
    #     plt.ylabel('SalePrice')
    #     fig = matplotlib.pyplot.gcf()
    #     fig.set_size_inches(18.5, 10.5)
    #     plt.savefig('plots/scatterplots/scatter_' + str(column) + '.png')
    #     plt.close()
    # exit()

    test_data = data_pipeline(test_data)
    test_data = feature_engineering(test_data)
    train_test_data = pd.concat([train_data,test_data])

    pred_set = pd.read_csv('test.csv')
    pred_set_id = pred_set['Id']
    pred_set = pred_set.drop(columns=dropped_attribs)
    pred_set = data_pipeline(pred_set)
    pred_set = feature_engineering(pred_set)

    scale_log_attribs = ['LotFrontage', 'MasVnrArea', 'LotArea',
                          '1stFlrSF','2ndFlrSF','BsmtFinSF1','BsmtFinSF2',
                          'BsmtUnfSF','GrLivArea',
                          'OpenPorchSF','TotalBsmtSF']

    for att in scale_log_attribs:
        try:
            train_test_data[att] = train_test_data[att].apply(np.log)
            train_test_data[att] = train_test_data[att].replace(-np.inf, 0)

            pred_set[att] = pred_set[att].apply(np.log)
            pred_set[att] = pred_set[att].replace(-np.inf, 0)
        except KeyError:
            print("Warning, KeyError, attrib not found in data")
            continue

    test_data = train_test_data[len(train_data):]
    train_data = train_test_data[:len(train_data)]


    # import seaborn as sns
    # for column in data:
    #     try:
    #         sns.distplot(data[column], color='blue', axlabel=column)
    #         fig = matplotlib.pyplot.gcf()
    #         fig.set_size_inches(18.5, 10.5)
    #         plt.savefig('plots/histograms/hist2_' + str(column) + '.png',dpi=100)
    #         plt.close()
    #     except:
    #         pass
    #
    # exit()

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

                #0.06
        if h > 0.05:
            idx = statCoef.index(h) - 1
            train_data = train_data.drop(columns=xlabels[idx])
            test_data = test_data.drop(columns=xlabels[idx])
            pred_set = pred_set.drop(columns=xlabels[idx])

        else:
            break

    stats.summary(ols, train_data, train_labels, xlabels=xlabels)

    #exit()
    pred1, vpred1 = linear_reg(train_data, train_labels, test_data, test_labels,pred_set,pred_set_id)
    pred2, vpred2 = lasso(train_data, train_labels, test_data, test_labels,pred_set,pred_set_id)
    pred3, vpred3 = ridge(train_data, train_labels, test_data, test_labels,pred_set,pred_set_id)
    pred4, vpred4 = bayesian_ridge(train_data, train_labels, test_data, test_labels,pred_set,pred_set_id)
    #pred5, vpred5 = gboost(train_data, train_labels, test_data, test_labels, pred_set, pred_set_id)

    pred = (pred1 + pred2 + pred3 + pred4) / 4
    vpred = (vpred1 + vpred2 + vpred3 + vpred4) / 4

    print("Combined Prediction")
    print("RMSLE ", mean_squared_log_error(test_labels, pred))
    print("RMSE ", mean_squared_error(test_labels, pred))
    print("Variance ", explained_variance_score(test_labels, pred))


    res = pd.DataFrame({"Id": pred_set_id, "SalePrice": vpred})
    res.to_csv("predictionsCombined.csv", index=False)

data = pd.read_csv('iowaHomes.csv')



data = data.drop(data[(data['OverallQual'] < 5.0) & (data['SalePrice'] > 200000)].index)
data = data.drop(data[(data['OverallQual'] == 8.0) & (data['SalePrice'] > 470000)].index)
data = data.drop(data[(data['OverallQual'] == 9.0) & (data['SalePrice'] > 430000)].index)

data = data.drop(data[(data['YearBuilt'] < 1960) & (data['SalePrice'] > 300000)].index)

data = data.drop(data[data['LotArea'] > 35000].index)

data = data.drop(data[(data['MSZoning'] == "RM") & (data['SalePrice'] > 300000)].index)

data = data.drop(data[(data['SalePrice'] > 550000)].index)


dropped_attribs = ["Id", "Alley",
                       "Street", 'PoolQC', 'Utilities', 'RoofStyle',
                       'RoofMatl', "PoolArea", 'BsmtFinSF1', 'BsmtFinSF2', 'GarageQual',
                        'Exterior2nd',
                       #'1stFlrSF', '2ndFlrSF',

                       #'YrSold','MoSold','SaleCondition','SaleType' #these are dropped because we can't
                       # estimate the price of a house based on these as the house hasn't been sold yet

                       ]
data = data.drop(columns=dropped_attribs)


main()
