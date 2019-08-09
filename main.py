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
import copy
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

    if save_model:
        filename = 'savedModels/ridge_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    return model


def feature_engineering(d, features):
    for feature in features:
        d[feature['name']] = 0
        for dep in feature['dependencies']:
            try:
                d[feature['name']] = d[feature['name']] + d[dep].astype(np.int)
            except ValueError:
                continue

    #d['TotalSF'] = d['TotalBsmtSF'].astype(np.int) + d['1stFlrSF'].astype(np.int) + d['2ndFlrSF'].astype(np.int)
    #d['PorchSF'] = d['OpenPorchSF'].astype(np.int) + d['EnclosedPorch'].astype(np.int) + d['3SsnPorch'].astype(np.int) + d['ScreenPorch'].astype(np.int)

    return d


def fill_missing_values(d):
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


def elem_in_array(a, data):
    for x in data:
        if x == a:
            return 1
    return 0


def run_prediction(models, pred_data_, load_models=True):

    for key in copy.deepcopy(pred_data_):
        if pred_data_[key] == "NULL":
            pred_data_.pop(key)

    featureOrder = pickle.load(open('iowaHomes/iowaHomes/predictionModels/training_data/featureOrder.sav', 'rb'))

    target_features = []
    for feat in featureOrder:
        target_features.append(feat)
        for e_feat in engineered_features:
            if e_feat['name'] == feat:
                target_features.remove(feat)
                for dep in e_feat['dependencies']:
                    target_features.append(dep)

    target_features_len = len(target_features)
    for feat in target_features[:]:
        if elem_in_array(feat, pred_data_) == 0:
            try:
                target_features.remove(feat)
                featureOrder.remove(feat)
            except ValueError:
                continue


    if target_features_len != len(target_features):
        load_models = False


    if load_models:
        t_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'iowaHomes/iowaHomes/predictionModels/training_data')
        train_test_data = pd.read_csv(os.path.join(t_data_path, 'training_data.csv'), keep_default_na=False)
    else:
        train_data, train_labels, train_test_data, test_data, test_labels = process_data()


    pred_data = pd.DataFrame(columns=list(pred_data_.keys()))
    pred_data.loc[0] = list(pred_data_.values())


    pred_data = pd.concat([train_test_data, pred_data])

    pred_data = feature_engineering(pred_data, engineered_features)

    pred_data = encode_values(pred_data)

    train_test_data = encode_values(train_test_data)
    pred_data = log_scale_values(pred_data, train_test_data)

    pred_data = pred_data[len(train_test_data):]

    pred_values = []
    for key in featureOrder:
        pred_values.append(pred_data[key].values[0])


    pred_data = pd.DataFrame(columns=featureOrder)
    pred_data.loc[0] = pred_values


    if load_models:
        models = load_models_from_file(models)
        pred = 0
        for model in models:
            pred = pred + model.predict(pred_data)

        pred = pred / len(models)
        #print(pred)
        #print(np.exp(pred))

    else:

        cols_to_drop = list(train_data.columns)
        for feat in featureOrder:
            try:
                cols_to_drop.remove(feat)

            except:
                pass

        train_data = train_data.drop(columns=cols_to_drop, errors="ignore")

        pred = 0
        for model in models:
            trained_m = model['func'](train_data, train_labels)
            pred = pred + trained_m.predict(pred_data)

        pred = pred / len(models)
        #print(pred)
        #print(np.exp(pred))
    return pred


def score_models(models, load_models=False):

    train_data, train_labels, train_test_data_, test_data, test_labels = process_data()

    if load_models:
        return 0
    else:
        for model in models:
            trained_m = model['func'](train_data, train_labels)
            model_score(model['name'], trained_m, test_data, test_labels)


def save_models_to_file(models):
    train_data, train_labels, train_test_data_, test_data, test_labels = process_data(save_train_data=True)

    for model in models:
        trained_m = model['func'](train_data, train_labels)
        filename = 'iowaHomes/iowaHomes/predictionModels/' + model['name']
        pickle.dump(trained_m, open(filename, 'wb'))
        print("Saved " + model['name'] + " to " + filename)


def load_models_from_file(models):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    loaded_models = []
    for model in models:
        path = os.path.join(BASE_DIR, 'IowaHomes/iowaHomes/iowaHomes/predictionModels/')
        path = os.path.join(path, model['name'])
        loaded_models.append(pickle.load(open(path, 'rb')))

    return loaded_models


def process_data(save_train_data=False):
    data = pd.read_csv('iowaHomes.csv')

    data = data.drop(columns='Id')
    data = drop_outliers(data)

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    train_labels = train_data['SalePrice'].apply(np.log)
    train_data = train_data.drop(columns='SalePrice')

    test_labels = test_data['SalePrice'].apply(np.log)
    test_data = test_data.drop(columns='SalePrice')

    train_data = fill_missing_values(train_data)
    test_data = fill_missing_values(test_data)

    train_data = feature_engineering(train_data, engineered_features)

    test_data = feature_engineering(test_data, engineered_features)

    train_test_data = pd.concat([train_data, test_data])  # data joined to perform factorization and scaling


    train_test_data_ = copy.deepcopy(train_test_data)

    if save_train_data:
        train_test_data.to_csv('iowaHomes/iowaHomes/predictionModels/training_data/training_data.csv',index=False)  # saving out data with no missing values to use when making predictions in production

    train_test_data = encode_values(train_test_data)

    train_test_data = log_scale_values(train_test_data, train_test_data)

    train_test_data = drop_columns(train_test_data)

    test_data = train_test_data[len(train_data):]
    train_data = train_test_data[:len(train_data)]


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

    featureOrder = []
    for x in range(len(statCoef)-1):
        featureOrder.append(xlabels[x])

    stats.summary(ols, train_data, train_labels, xlabels=xlabels)

    if save_train_data:
        pickle.dump(featureOrder, open('iowaHomes/iowaHomes/predictionModels/training_data/featureOrder.sav', 'wb'))

    return train_data, train_labels, train_test_data_,test_data, test_labels


models = [    {'name': 'LinearRegression.sav',
               'func': linear_reg},
              {'name': 'Lasso.sav',
               'func': lasso},
              {'name': 'BayesianRidge.sav',
               'func': bayesian_ridge},
              {'name': 'Ridge.sav',
               'func': ridge},
              ]

engineered_features = [{"name": "TotalSF",
                            "dependencies": ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']},
                           {"name": "PorchSF",
                            "dependencies": ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']}
                           ]


dropped_attribs = ["Alley",
                   "Street", 'PoolQC', 'Utilities', 'RoofStyle',
                   'RoofMatl', "PoolArea", 'BsmtFinSF1', 'BsmtFinSF2', 'GarageQual',
                   'Exterior2nd',

                   'BsmtFinType2',
                   'YrSold', 'MoSold', 'SaleCondition', 'SaleType'  # these are dropped because we can't
                   # estimate the price of a house based on these as the house hasn't been sold yet

                   ]

for feat in engineered_features:
    for dep in feat['dependencies']:
        dropped_attribs.append(dep)



if __name__ == "__main__":
    pass

