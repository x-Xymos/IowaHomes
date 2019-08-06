import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
import pickle, warnings, os
from regressors import stats
import cProfile

warnings.filterwarnings("ignore", category=FutureWarning,)
warnings.filterwarnings("ignore", category=UserWarning,)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None


def linear_reg(train_data, train_labels, test_data, test_labels):

    reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1,
                     normalize=False)
    reg.fit(train_data, train_labels)

    filename = 'savedModels/l_reg_model.sav'
    pickle.dump(reg, open(filename, 'wb'))




    pred = reg.predict(test_data)
    print("Linear Regression")
    print("RMSLE ", mean_squared_log_error(test_labels, pred))
    print("RMSE ", mean_squared_error(test_labels, pred))
    print("Variance ", explained_variance_score(test_labels, pred))

    return pred


def lasso(train_data, train_labels, test_data, test_labels):

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

    filename = 'savedModels/lasso_model.sav'
    pickle.dump(reg, open(filename, 'wb'))

    pred = reg.predict(test_data)
    print("Lasso")
    print("RMSLE ", mean_squared_log_error(test_labels, pred))
    print("RMSE ", mean_squared_error(test_labels, pred))
    print("Variance ", explained_variance_score(test_labels, pred))


    return pred


def bayesian_ridge(train_data, train_labels, test_data, test_labels):

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
    filename = 'savedModels/b_ridge_model.sav'
    pickle.dump(reg, open(filename, 'wb'))

    pred = reg.predict(test_data)
    print("Bayesian Ridge")
    print("RMSLE ",mean_squared_log_error(test_labels, pred))
    print("RMSE ",mean_squared_error(test_labels, pred))
    print("Variance ",explained_variance_score(test_labels, pred))



    return pred
    ##########################################################################


def ridge(train_data, train_labels, test_data, test_labels):
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
    filename = 'savedModels/ridge_model.sav'
    pickle.dump(reg, open(filename, 'wb'))

    pred = reg.predict(test_data)
    print("Ridge")
    print("RMSLE ",mean_squared_log_error(test_labels, pred))
    print("RMSE ",mean_squared_error(test_labels, pred))
    print("Variance ",explained_variance_score(test_labels, pred))

    return pred
    ##########################################################################


def feature_engineering(d):
    d['TotalSF'] = d['TotalBsmtSF'].astype(np.int) + d['1stFlrSF'].astype(np.int) + d['2ndFlrSF'].astype(np.int)
    #d = d.drop(columns=['1stFlrSF', '2ndFlrSF','TotalBsmtSF'])

    #d['Bathrooms'] = d['BsmtFullBath'] + d['BsmtHalfBath'] + d['FullBath'] + d['HalfBath']
    #d = d.drop(columns=['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'])

    d['PorchSF'] = d['OpenPorchSF'].astype(np.int) + d['EnclosedPorch'].astype(np.int) + d['3SsnPorch'].astype(np.int) + d['ScreenPorch'].astype(np.int)
    #d = d.drop(columns=['OpenPorchSF', 'EnclosedPorch','3SsnPorch','ScreenPorch'])

    return d


def fill_missing_val(d):
    custom_cat_attribs = ['FireplaceQu', "PoolQC", 'BsmtQual', 'BsmtCond',
                          'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                          'GarageType', 'GarageQual', 'GarageFinish',
                          'Fence', 'MiscFeature', 'GarageCond']
    naImputer = SimpleImputer(strategy="constant", fill_value="NA")
    for att in custom_cat_attribs:
        try:
            d[att] = naImputer.fit_transform(d[att].values.reshape(-1, 1))
        except KeyError:
            continue

    cat_attribs = list(d.select_dtypes(include=[np.object]).columns)
    freqImputer = SimpleImputer(strategy="most_frequent")
    for att in cat_attribs:
        try:
            d[att] = freqImputer.fit_transform(d[att].values.reshape(-1, 1))
        except KeyError:
            continue

    num_attribs = list(d.select_dtypes(include=[np.number]).columns)
    numImputer = SimpleImputer(strategy="median")
    for att in num_attribs:
        try:
            d[att] = numImputer.fit_transform(d[att].values.reshape(-1, 1))
        except KeyError:
            continue



    return d


def encode_values(d):

    factorization_attribs = list(d.select_dtypes(include=[np.object]).columns)
    factorization_attribs = factorization_attribs + ['YearBuilt', 'YearRemodAdd',
                                                     'GarageYrBlt']


    d['YearBuilt'] = d['YearBuilt'].astype(np.int)
    d['YearRemodAdd'] = d['YearRemodAdd'].astype(np.int)
    d['GarageYrBlt'] = d['GarageYrBlt'].astype(np.int)


    enc = OrdinalEncoder()
    for att in factorization_attribs:
        try:
            d[att] = enc.fit_transform(d[att].values.reshape(-1, 1))
        except:
            continue

    return d


def log_scale_values(d):

    scale_log_attribs = ['LotFrontage', 'MasVnrArea', 'LotArea',
                         '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2',
                         'BsmtUnfSF', 'GrLivArea',
                         'OpenPorchSF', 'TotalBsmtSF']

    for att in scale_log_attribs:
        try:
            d[att] = d[att].astype(float).apply(np.log)
            d[att] = d[att].astype(float).replace(-np.inf, 0)
        except:
            continue

    return d


def drop_colums(d):
    dropped_attribs = ["Alley",
                       "Street", 'PoolQC', 'Utilities', 'RoofStyle',
                       'RoofMatl', "PoolArea", 'BsmtFinSF1', 'BsmtFinSF2', 'GarageQual',
                       'Exterior2nd',

                       '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF',
                       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',

                       'BsmtFinType2',
                       'YrSold', 'MoSold', 'SaleCondition', 'SaleType'  # these are dropped because we can't
                       # estimate the price of a house based on these as the house hasn't been sold yet

                       ]
    d = d.drop(columns=dropped_attribs)

    return d


def run_prediction(args):

    args_df = pd.DataFrame(columns=list(args.keys()))
    args_df.loc[0] = list(args.values())

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'savedModels/28features')


    training_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'savedModels/training_data.csv'), keep_default_na=False)

    args_df = pd.concat([training_data, args_df])


    args_df = args_df.reset_index(drop=True)

    args_df = feature_engineering(args_df)
    args_df = drop_colums(args_df)

    args_df = args_df.dropna(axis=1)

    args_df = encode_values(args_df)


    args_df = log_scale_values(args_df)
    args_df = args_df[len(training_data):]

    n_models = 0
    preds = 0
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            reg = pickle.load(open(os.path.join(path, file), 'rb'))
            pred = reg.predict(args_df)
            print(pred)
            preds = preds + pred
            n_models += 1


    preds = preds / n_models
    print(preds)
    print(np.exp(preds))
    import math
    print(math.exp(preds))

    return 0



def main():

    data = pd.read_csv('iowaHomes.csv')

    data = data.drop(columns='Id')

    data = data.drop(data[(data['OverallQual'] < 5.0) & (data['SalePrice'] > 200000)].index)
    data = data.drop(data[(data['OverallQual'] == 8.0) & (data['SalePrice'] > 470000)].index)
    data = data.drop(data[(data['OverallQual'] == 9.0) & (data['SalePrice'] > 430000)].index)

    data = data.drop(data[(data['YearBuilt'] < 1960) & (data['SalePrice'] > 300000)].index)

    data = data.drop(data[data['LotArea'] > 35000].index)

    data = data.drop(data[(data['MSZoning'] == "RM") & (data['SalePrice'] > 300000)].index)

    data = data.drop(data[(data['SalePrice'] > 550000)].index)



    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    train_labels = train_data['SalePrice'].apply(np.log)
    train_data = train_data.drop(columns='SalePrice')


    test_labels = test_data['SalePrice'].apply(np.log)
    test_data = test_data.drop(columns='SalePrice')

    train_data = fill_missing_val(train_data)
    test_data = fill_missing_val(test_data)

    train_test_data = pd.concat([train_data, test_data])
    train_test_data.to_csv('savedModels/training_data.csv', index=False) #saving out data with no missing values to use when making predictions in production


    train_data = feature_engineering(train_data)
    test_data = feature_engineering(test_data)


    train_test_data = pd.concat([train_data, test_data]) #data joined to perform factorization and scaling


    train_test_data = drop_colums(train_test_data)

    train_test_data = encode_values(train_test_data)
    train_test_data = log_scale_values(train_test_data)


    test_data = train_test_data[len(train_data):]
    train_data = train_test_data[:len(train_data)]

    while True:

        ols = linear_model.LinearRegression()

        ols = ols.fit(train_data, train_labels)

        xlabels = list(train_data.columns)
        statCoef = list(stats.coef_pval(ols, train_data,train_labels))

        h = 0
        for x in range(len(statCoef)-1):
            if h < statCoef[x+1]:
                h = statCoef[x+1]

        if h > 0.05:
            idx = statCoef.index(h) - 1
            train_data = train_data.drop(columns=xlabels[idx])
            test_data = test_data.drop(columns=xlabels[idx])

        else:
            break

    stats.summary(ols, train_data, train_labels, xlabels=xlabels)

    pred1 = linear_reg(train_data, train_labels, test_data, test_labels)
    pred2 = lasso(train_data, train_labels, test_data, test_labels)
    pred3 = ridge(train_data, train_labels, test_data, test_labels)
    pred4 = bayesian_ridge(train_data, train_labels, test_data, test_labels)


    pred = (pred1 + pred2 + pred3 + pred4) / 4

    print("Combined Prediction")
    print("RMSLE ", mean_squared_log_error(test_labels, pred))
    print("RMSE ", mean_squared_error(test_labels, pred))
    print("Variance ", explained_variance_score(test_labels, pred))

    print(pred)

if __name__ == "__main__":
    args = {'MSZoning': 'C',
            'OverallQual': '7',
            'OverallCond': '5',
            'ExterCond': 'TA',
            'YearBuilt': '2003',
            'YearRemodAdd': '2003',
            'LotArea': '8450',
            'GrLivArea': '1710',
            '1stFlrSF': '856',
            '2ndFlrSF': '854',
            'BsmtQual': 'Gd',
            'BsmtExposure': 'No',
            'BsmtFinType1': 'GLQ',
            'BsmtFullBath': '1',
            'TotalBsmtSF': '856',
            'HeatingQC': 'Ex',
            'CentralAir': 'Y',
            'BedroomAbvGr': '3',
            'KitchenAbvGr': '1',
            'KitchenQual': 'Gd',
            'Functional': 'Typ',
            'Fireplaces': '0',
            'GarageYrBlt': '2003',
            'GarageCars': '2',
            'GarageArea': '548',
            'PavedDrive': 'Y',
            'WoodDeckSF': '0',
            'OpenPorchSF': '61',
            'EnclosedPorch': '0',
            '3SsnPorch': '0',
            'ScreenPorch': '0',
            'MasVnrType': 'BrkFace',
            'MasVnrArea': '196'}

    #pr = cProfile.Profile()
    #pr.enable()

    #main()

    #pr.disable()
    #pr.print_stats(sort='time')
    print(run_prediction(args))
