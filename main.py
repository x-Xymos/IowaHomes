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


def linear_reg(train_data, train_labels, save_model=False):

    model = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1,
                     normalize=False)

    model.fit(train_data, train_labels)

    if save_model:
        filename = 'savedModels/l_reg_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    return model


def lasso(train_data, train_labels, save_model=False):

    model = linear_model.LassoCV(tol=0.00, precompute=True, normalize=True,
                             max_iter=300,fit_intercept=True, cv=13,
                             random_state=42)


    model.fit(train_data, train_labels)

    if save_model:
        filename = 'savedModels/lasso_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    return model


def bayesian_ridge(train_data, train_labels, save_model=False):

    model = linear_model.BayesianRidge(tol=18.79999999999995, normalize=True,
                                     n_iter=245, lambda_2=191.0, lambda_1=240.0,
                                     fit_intercept=True, alpha_2=0.026000000000000002,
                                     alpha_1=13.39999999999997)

    model.fit(train_data, train_labels)

    if save_model:
        filename = 'savedModels/b_ridge_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    return model


def ridge(train_data, train_labels, save_model=False):

    model = linear_model.Ridge(tol=0.06, solver="svd", max_iter=4500, fit_intercept=True,
                             alpha=0.46, random_state=42)

    model.fit(train_data, train_labels)

    return model


def save_models(models, train_data, train_labels):

    for model in models:
        trained_m = model['func'](train_data, train_labels)
        filename = 'iowaHomes/iowaHomes/predictionModels/' + model['name'] + ".sav"
        pickle.dump(trained_m, open(filename, 'wb'))
        print("Saved " + model['name'] + "to " + filename)


def load_models(models):
    # for model in models:
    #     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #     path = os.path.join(BASE_DIR, 'iowaHomes/predictionModels/')
    #     reg = pickle.load(open(path + model, 'rb'))

    return 0


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
            print(att)
            continue

    return d


def log_scale_values(d, training_data):

    from sklearn.preprocessing import FunctionTransformer
    transformer = FunctionTransformer(np.log1p, validate=True)
    transformer.fit(training_data)

    scale_log_attribs = ['LotFrontage', 'MasVnrArea', 'LotArea',
                         '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2',
                         'BsmtUnfSF', 'GrLivArea',
                         'OpenPorchSF', 'TotalBsmtSF']

    for att in scale_log_attribs:
        try:
            d[att] = transformer.transform(d[att].values.reshape(-1, 1))

            #d[att] = d[att].astype(float).apply(np.log)
            #d[att] = d[att].astype(float).replace(-np.inf, 0)
        except:
            continue

    return d


def drop_columns(d):
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
    d = d.drop(columns=dropped_attribs, errors="ignore")

    return d


def drop_outliers(d):
    d = d.drop(d[(d['OverallQual'] < 5.0) & (d['SalePrice'] > 200000)].index)
    d = d.drop(d[(d['OverallQual'] == 8.0) & (d['SalePrice'] > 470000)].index)
    d = d.drop(d[(d['OverallQual'] == 9.0) & (d['SalePrice'] > 430000)].index)

    d = d.drop(d[(d['YearBuilt'] < 1960) & (d['SalePrice'] > 300000)].index)

    d = d.drop(d[d['LotArea'] > 35000].index)

    d = d.drop(d[(d['MSZoning'] == "RM") & (d['SalePrice'] > 300000)].index)

    d = d.drop(d[(d['SalePrice'] > 550000)].index)
    
    return d


def model_score(name, model, test_data, test_labels):

    pred = model.predict(test_data)
    print(name)
    print("RMSLE ", mean_squared_log_error(test_labels, pred))
    print("RMSE ", mean_squared_error(test_labels, pred))
    print("Variance ", explained_variance_score(test_labels, pred))

def score_models(models, train_data, train_labels, test_data, test_labels, load_models=False):

    if load_models:
        return 0
    else:
        for model in models:
            trained_m = model['func'](train_data, train_labels)
            model_score(model['name'],trained_m, test_data, test_labels)


def process_data(pred_data=None):
    data = pd.read_csv('iowaHomes.csv')

    data = data.drop(columns='Id')

    data = drop_outliers(data)

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    if pred_data is not None:

        pred_data_ = pd.DataFrame(columns=list(pred_data.keys()))
        pred_data_.loc[0] = list(pred_data.values())
        test_data = pd.concat([test_data, pred_data_])
        print(test_data.tail(n=2))

    train_labels = train_data['SalePrice'].apply(np.log)
    train_data = train_data.drop(columns='SalePrice')

    test_labels = test_data['SalePrice'].apply(np.log)

    if pred_data is not None:
        test_labels.iloc[test_labels.shape[0]-1] = 10


    test_data = test_data.drop(columns='SalePrice')

    train_data = fill_missing_val(train_data)
    test_data = fill_missing_val(test_data)

    train_data = feature_engineering(train_data)
    test_data = feature_engineering(test_data)

    train_test_data = pd.concat([train_data, test_data])  # data joined to perform factorization and scaling

    if pred_data is None:
        train_test_data.to_csv('iowaHomes/iowaHomes/predictionModels/training_data/training_data.csv',
                               index=False)  # saving out data with no missing values to use when making predictions in production

    train_test_data = encode_values(train_test_data)

    train_test_data = log_scale_values(train_test_data, train_test_data)
    train_test_data = drop_columns(train_test_data)

    test_data = train_test_data[len(train_data):]
    train_data = train_test_data[:len(train_data)]


    if pred_data is None:
        droppedCols = []
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
                droppedCols.append(xlabels[idx])
                train_data = train_data.drop(columns=xlabels[idx])
                test_data = test_data.drop(columns=xlabels[idx])

            else:
                break

        filename = 'iowaHomes/iowaHomes/predictionModels/training_data/droppedCols.sav'
        pickle.dump(droppedCols, open(filename, 'wb'))

        stats.summary(ols, train_data, train_labels, xlabels=xlabels)

        print(test_data.tail(n=2))
        return train_data, train_labels, test_data, test_labels

    else:
        return test_data.iloc[test_data.shape[0]-1]


def run_prediction(models, pred_data_, train_data, train_labels, load_models=False):

    pred_data = {
        'MSSubClass': 20.0,
        'MSZoning': 'RL',
        'OverallQual': 5,
        'OverallCond': 6,
        'BsmtUnfSF': 0.0,
        'ExterCond': 'TA',
        'GarageCond': 'TA',
        'LotFrontage': 80.0,
        'TotRmsAbvGrd': 7.0,
        'YearBuilt': 1977,
        'YearRemodAdd': 1977,
        'LotArea': 12984,
        'GrLivArea': 1647,
        '1stFlrSF': 1647,
        '2ndFlrSF': 0,
        'BsmtQual': 'Gd',
        'BsmtExposure': 'Mn',
        'BsmtFinType1': 'ALQ',
        'BsmtFullBath': 1,
        'TotalBsmtSF': 1430,
        'HeatingQC': 'Ex',
        'CentralAir': 'Y',
        'BedroomAbvGr': 3,
        'KitchenAbvGr': 1,
        'KitchenQual': 'Gd',
        'Functional': 'Typ',
        'Fireplaces': 1,
        'GarageYrBlt': 1977,
        'GarageCars': 2,
        'GarageArea': 621,
        'PavedDrive': 'Y',
        'WoodDeckSF': 0,
        'OpenPorchSF': 0,
        'EnclosedPorch': 0,
        '3SsnPorch': 0,
        'ScreenPorch': 0,
        'MasVnrType': 'BrkFace',
        'MasVnrArea': 459}
    cols = ["MSZoning" ,  "LotArea",  "OverallQual" ,
     "OverallCond" , "YearBuilt" , "YearRemodAdd"  ,
     "MasVnrType" , "MasVnrArea" , "ExterCond" ,
     "BsmtQual" , "BsmtExposure" , "BsmtFinType1" ,
     "HeatingQC" , "CentralAir" , "GrLivArea",
     "BsmtFullBath"  ,"BedroomAbvGr" , "KitchenAbvGr" ,
     "KitchenQual" , "Functional" , "Fireplaces"  ,
     "GarageYrBlt" , "GarageCars" , "GarageArea" ,
     "PavedDrive",  "WoodDeckSF" , "TotalSF" ,
     "PorchSF"]
    pred_data = process_data(pred_data)

    pred_values = []
    for key in cols:
        pred_values.append(pred_data[key])

    pred_data = pd.DataFrame(columns=cols)
    pred_data.loc[0] = pred_values


   #  pred_data = pd.DataFrame(columns=list(pred_data_.keys()))
   #  pred_data.loc[0] = list(pred_data_.values())
   #
   #  path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iowaHomes/iowaHomes/predictionModels/training_data')
   #
   #  training_data = pd.read_csv(os.path.join(path, 'training_data.csv'), keep_default_na=False)
   #
   #
   #  pred_data = pd.concat([training_data, pred_data])
   #
   #  pred_data = feature_engineering(pred_data)
   #
   #  pred_data = pred_data.dropna(axis=1)
   #
   #  pred_data = encode_values(pred_data)
   #
   #
   #  training_data = encode_values(training_data)
   #  pred_data = log_scale_values(pred_data, pred_data)
   #
   #
   #
   #  pred_data = drop_columns(pred_data)
   #  pred_data = pred_data[len(training_data):]
   #
    droppedCols = pickle.load(open('iowaHomes/iowaHomes/predictionModels/training_data/droppedCols.sav', 'rb'))

    pred_data = pred_data.drop(columns=droppedCols, errors="ignore")

    print(pred_data)
   #  print("-----------")
   #  print("pred_data")
   #  for col in cols:
   #      print(pred_data[col].tail(n=1))
   # # print(pred_data)
    if load_models:
        return 0
    else:
        pred = 0
        for model in models:
            trained_m = model['func'](train_data, train_labels)
            pred = pred + trained_m.predict(pred_data)

    pred = pred / len(models)
    print(pred)
    print(np.exp(pred))
    print(train_labels.tail(n=1))
    train_labels = np.exp(train_labels)
    print(train_labels.tail(n=1))
    exit()


    n_models = 0
    preds = 0
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            reg = pickle.load(open(os.path.join(path, file), 'rb'))
            pred = reg.predict(args_df)
            print(pred)
            preds = preds + pred
            n_models += 1


    preds = preds[0] / n_models

    preds = log_scale_values(preds, args_df_copy)

    print(preds)
    #print(np.exp(preds))
    #import math
    #print(math.exp(preds))

    return 0


def main():
    args = {
        'MSSubClass': 20.0,
        'MSZoning': 'RL',
        'OverallQual': 5,
        'OverallCond': 6,
        'BsmtUnfSF': 0.0,
        'ExterCond': 'TA',
        'GarageCond': 'TA',
        'LotFrontage': 80.0,
        'TotRmsAbvGrd': 7.0,
        'YearBuilt': 1977,
        'YearRemodAdd': 1977,
        'LotArea': 12984,
        'GrLivArea': 1647,
        '1stFlrSF': 1647,
        '2ndFlrSF': 0,
        'BsmtQual': 'Gd',
        'BsmtExposure': 'Mn',
        'BsmtFinType1': 'ALQ',
        'BsmtFullBath': 1,
        'TotalBsmtSF': 1430,
        'HeatingQC': 'Ex',
        'CentralAir': 'Y',
        'BedroomAbvGr': 3,
        'KitchenAbvGr': 1,
        'KitchenQual': 'Gd',
        'Functional': 'Typ',
        'Fireplaces': 1,
        'GarageYrBlt': 1977,
        'GarageCars': 2,
        'GarageArea': 621,
        'PavedDrive': 'Y',
        'WoodDeckSF': 0,
        'OpenPorchSF': 0,
        'EnclosedPorch': 0,
        '3SsnPorch': 0,
        'ScreenPorch': 0,
        'MasVnrType': 'BrkFace',
        'MasVnrArea': 459}

    models = [{'name': 'LinearRegression',
               'func': linear_reg},
              {'name': 'Lasso',
               'func': lasso},
              {'name': 'BayesianRidge',
               'func': bayesian_ridge},
              {'name': 'Ridge',
               'func': ridge},
              ]

    train_data, train_labels, test_data, test_labels = process_data()



    #score_models(models,train_data,train_labels, test_data, test_labels)


    #save_models(models,train_data,train_labels)

    droppedCols = pickle.load(open('iowaHomes/iowaHomes/predictionModels/training_data/droppedCols.sav', 'rb'))
    #print(test_data.drop(columns=droppedCols, errors="ignore").tail(n=1))


    run_prediction(models, args,train_data, train_labels)



    #print(test_data.tail(n=2))
    #exit()
    # training_data = pd.read_csv(
    #     os.path.join(os.path.dirname(os.path.abspath(__file__)), 'savedModels/training_data.csv'),
    #     keep_default_na=False)


cols = [ "BedroomAbvGr" , "BsmtExposure",  "BsmtFinType1",
         "BsmtFullBath",  "BsmtQual" , "CentralAir",
         "ExterCond",  "Fireplaces",  "Functional",
         "GarageArea",  "GarageCars",  "GarageYrBlt",
         "GrLivArea",  "HeatingQC",  "KitchenAbvGr",
         "KitchenQual",   "LotArea",  "MSZoning",
         "MasVnrArea",  "MasVnrType",  "OverallCond" ,
         "OverallQual" , "PavedDrive"  ,"PorchSF"  ,
         "TotalSF" , "WoodDeckSF",  "YearBuilt" , "YearRemodAdd"]

if __name__ == "__main__":
    # args = {'MSZoning': 'RH',
    #         'OverallQual': 5,
    #         'OverallCond': 6,
    #         'ExterCond': 'TA',
    #         'YearBuilt': 961,
    #         'YearRemodAdd': 1961,
    #         'LotArea': 11622,
    #         'GrLivArea': 896,
    #         '1stFlrSF': 896,
    #         '2ndFlrSF': 0,
    #         'BsmtQual': 'TA',
    #         'BsmtExposure': 'No',
    #         'BsmtFinType1': 'Rec',
    #         'BsmtFullBath': 0,
    #         'TotalBsmtSF': 882,
    #         'HeatingQC': 'TA',
    #         'CentralAir': 'Y',
    #         'BedroomAbvGr': 2,
    #         'KitchenAbvGr': 1,
    #         'KitchenQual': 'TA',
    #         'Functional': 'Typ',
    #         'Fireplaces': 0,
    #         'GarageYrBlt': 1961,
    #         'GarageCars': 1,
    #         'GarageArea': 730,
    #         'PavedDrive': 'Y',
    #         'WoodDeckSF': 40,
    #         'OpenPorchSF': 0,
    #         'EnclosedPorch': 0,
    #         '3SsnPorch': 0,
    #         'ScreenPorch': 120,
    #         'MasVnrType': 'None',
    #         'MasVnrArea': 0}
    # #pr = cProfile.Profile()
    # #pr.enable()
    #

    # args_df = pd.DataFrame(columns=list(args.keys()))
    # args_df.loc[0] = list(args.values())
    # test_data = pd.concat([test_data, args_df])
    # print(test_data.tail(n=2))

    main()

    #pr.disable()
    #pr.print_stats(sort='time')
    #print(run_prediction(args))
