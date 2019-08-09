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
from sklearn.preprocessing import FunctionTransformer

warnings.filterwarnings("ignore", category=FutureWarning,)
warnings.filterwarnings("ignore", category=UserWarning,)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None


def linear_reg(train_data, train_labels):
    """

            Fits the train_data and train_labels to the model returns it


            Paramaters
            ----------
            train_data : dataframe
            train_labels : dataframe

            Returns
            ----------
            model : linear_reg

        """

    model = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1,
                     normalize=False)

    model.fit(train_data, train_labels)

    return model


def lasso(train_data, train_labels):
    """

            Fits the train_data and train_labels to the model returns it


            Paramaters
            ----------
            train_data : dataframe
            train_labels : dataframe

            Returns
            ----------
            model : lasso

        """

    model = linear_model.LassoCV(tol=0.00, precompute=True, normalize=True,
                             max_iter=300,fit_intercept=True, cv=13,
                             random_state=42)


    model.fit(train_data, train_labels)

    return model


def bayesian_ridge(train_data, train_labels):
    """

         Fits the train_data and train_labels to the model returns it


         Paramaters
         ----------
         train_data : dataframe
         train_labels : dataframe

         Returns
         ----------
         model : bayesian_ridge

     """

    model = linear_model.BayesianRidge(tol=18.79999999999995, normalize=True,
                                     n_iter=245, lambda_2=191.0, lambda_1=240.0,
                                     fit_intercept=True, alpha_2=0.026000000000000002,
                                     alpha_1=13.39999999999997)

    model.fit(train_data, train_labels)

    return model


def ridge(train_data, train_labels):
    """

       Fits the train_data and train_labels to the model returns it


       Paramaters
       ----------
       train_data : dataframe
       train_labels : dataframe

       Returns
       ----------
       model : Ridge

   """

    model = linear_model.Ridge(tol=0.06, solver="svd", max_iter=4500, fit_intercept=True,
                             alpha=0.46, random_state=42)

    model.fit(train_data, train_labels)

    return model


def feature_engineering(d):
    """
            Creates new features in d based on the dependencies and features in engineered_features


            Paramaters
            ----------
            d : dataframe

            Returns
            ----------
            d : dataframe

        """

    for feature in engineered_features:
        d[feature['name']] = 0
        for dep in feature['dependencies']:
            try:
                d[feature['name']] = d[feature['name']] + d[dep].astype(np.int)
            except ValueError:
                continue

    return d


def fill_missing_values(d):
    """
        Applies a number of Imputers to fill in missing values


        Paramaters
        ----------
        d : dataframe

        Returns
        ----------
        d : dataframe

    """

    custom_cat_features = ['FireplaceQu', "PoolQC", 'BsmtQual', 'BsmtCond',
                          'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                          'GarageType', 'GarageQual', 'GarageFinish',
                          'Fence', 'MiscFeature', 'GarageCond']
    naImputer = SimpleImputer(strategy="constant", fill_value="NA")
    for att in custom_cat_features:
        try:
            d[att] = naImputer.fit_transform(d[att].values.reshape(-1, 1))
        except KeyError:
            continue

    cat_features = list(d.select_dtypes(include=[np.object]).columns)
    freqImputer = SimpleImputer(strategy="most_frequent")
    for att in cat_features:
        try:
            d[att] = freqImputer.fit_transform(d[att].values.reshape(-1, 1))
        except KeyError:
            continue

    num_features = list(d.select_dtypes(include=[np.number]).columns)
    numImputer = SimpleImputer(strategy="median")
    for att in num_features:
        try:
            d[att] = numImputer.fit_transform(d[att].values.reshape(-1, 1))
        except KeyError:
            continue



    return d


def encode_values(d):
    """
         Applies an ordinal encoder to d dataframe based on the columns
         defined in factorization_features


         Paramaters
         ----------
         d : dataframe

         Returns
         ----------
         d : dataframe

     """

    factorization_features = list(d.select_dtypes(include=[np.object]).columns)
    factorization_features = factorization_features + ['YearBuilt', 'YearRemodAdd',
                                                     'GarageYrBlt']


    enc = OrdinalEncoder()
    for att in factorization_features:
        try:
            d[att] = enc.fit_transform(d[att].values.reshape(-1, 1))
        except:
            continue

    return d


def log_scale_values(d, training_data):
    """

         Fits training_data to a FunctionTransformer and then applies that
         FunctionTransformer to the d dataframe on the columns defined
         in scale_log_features


         Paramaters
         ----------
         d : dataframe
         training_data : dataframe

         Returns
         ----------
         d : dataframe

     """


    transformer = FunctionTransformer(np.log1p, validate=True)
    transformer.fit(training_data)

    scale_log_features = ['LotFrontage', 'MasVnrArea', 'LotArea',
                         '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2',
                         'BsmtUnfSF', 'GrLivArea',
                         'OpenPorchSF', 'TotalBsmtSF']

    for att in scale_log_features:
        try:
            d[att] = transformer.transform(d[att].values.reshape(-1, 1))

        except:
            continue

    return d


def drop_columns(d):
    """

          Takes in input dataframe and drops the columns defined in dropped_features

          Paramaters
          ----------
          d : dataframe

          Returns
          ----------
          d : dataframe

      """

    d = d.drop(columns=dropped_features, errors="ignore")

    return d


def drop_outliers(d):
    """

        Takes in input dataframe and drops the specified rows

        Paramaters
        ----------
        d : dataframe

        Returns
        ----------
        d : dataframe

        """

    d = d.drop(d[(d['OverallQual'] < 5.0) & (d['SalePrice'] > 200000)].index)
    d = d.drop(d[(d['OverallQual'] == 8.0) & (d['SalePrice'] > 470000)].index)
    d = d.drop(d[(d['OverallQual'] == 9.0) & (d['SalePrice'] > 430000)].index)

    d = d.drop(d[(d['YearBuilt'] < 1960) & (d['SalePrice'] > 300000)].index)

    d = d.drop(d[d['LotArea'] > 35000].index)

    d = d.drop(d[(d['MSZoning'] == "RM") & (d['SalePrice'] > 300000)].index)

    d = d.drop(d[(d['SalePrice'] > 550000)].index)
    
    return d


def elem_in_array(elem, data):

    """

    Loops through data to check if it contains elem

    Paramaters
    ----------
    elem : any


    data : any



    Returns
    ----------
    0 : if elem wasn't found in data
    1 : if elem was found in data

    """

    for x in data:
        if x == elem:
            return 1
    return 0


def run_prediction(models, user_input_, load_models=True):

    """

    Run prediction on a single row of data

    Parameters
    ----------
    models : list
            a list of model dictionaries containing the name of the filename
            and the function name

            Example: [ {'name': 'LinearRegression.sav','func': linear_reg},
                        {'name': 'Lasso.sav','func': lasso} ]

   user_input_ : dictionary
                a dictionary of feature names and their passed values
                which are turned into a single row of data to run the prediction on

            Example: {'MSSubClass': '20.0',
                    'MSZoning': 'RL',
                    'OverallQual': '5',
                    'OverallCond': '6',
                    'BsmtUnfSF': '0.0',
                    'ExterCond': 'TA'}

    load_models : boolean, optional, default True
                can be overridden to False to force the function to always retrain the model

    Returns
    ----------
    A prediction number

    """



    for key in copy.deepcopy(user_input_):
        #checking user input data and removing null values
        if user_input_[key] == "":
            user_input_.pop(key)

    if len(user_input_) == 0:
        #if user_input is empty return
        return 0

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FOPATH = os.path.join(BASE_DIR, 'predictModel/predictionModels/training_data/featureOrder.sav')
    featureOrder = pickle.load(open(FOPATH, 'rb'))

    target_features = []
    # converting features that were engineered, into their dependencies
    for feat in featureOrder:
        target_features.append(feat)
        for e_feat in engineered_features:
            if e_feat['name'] == feat:
                target_features.remove(feat)
                for dep in e_feat['dependencies']:
                    target_features.append(dep)


    target_features_len = len(target_features)
    # comparing user_input to the target_features to see if
    # any features the model was trained with are missing
    for feat in target_features[:]:
        if elem_in_array(feat, user_input_) == 0:
            try:
                target_features.remove(feat)
                featureOrder.remove(feat)
            except ValueError:
                continue


    if target_features_len != len(target_features):
        # if there are any missing features then we
        # don't load the saved model and instead retrain the model
        load_models = False


    if load_models:
        t_data_path = os.path.join(BASE_DIR, 'predictModel/predictionModels/training_data/training_data.csv')
        train_test_data = pd.read_csv(t_data_path, keep_default_na=False)

    else:
        train_data, train_labels, train_test_data, test_data, test_labels = process_data()

    import re
    for key in list(train_test_data.columns):
        # converting/matching user input data to training data dtypes
        key_dtype = re.sub(r'\d+', '',  type(train_test_data[key][0]).__name__)
        try:
            if key_dtype == "int":
                user_input_[key] = int(user_input_[key])
            elif key_dtype == "float":
                user_input_[key] = float(user_input_[key])
        except:
            continue

    # converting the user input into a 1 row dataframe
    user_input = pd.DataFrame(columns=list(user_input_.keys()))
    user_input.loc[0] = list(user_input_.values())

    # combining the saved out train_test_data with the user_input so we
    # can apply the same pipeline to the user input we applied to the training data
    user_input = pd.concat([train_test_data, user_input])
    user_input = feature_engineering(user_input)
    user_input = encode_values(user_input)

    train_test_data = encode_values(train_test_data)
    user_input = log_scale_values(user_input, train_test_data)

    user_input = user_input[len(train_test_data):]

    pred_values = []
    for key in featureOrder:
        # retrieving only the values of features in user_input found in feature order
        pred_values.append(user_input[key].values[0])

    user_input = pd.DataFrame(columns=featureOrder)
    user_input.loc[0] = pred_values

    print(user_input)

    if load_models:
        models = load_models_from_file(models)
        pred = 0
        for model in models:
            pred = pred + model.predict(user_input)

        pred = pred / len(models)
        return np.exp(pred)[0]

    else:

        # if any required features are missing from the
        # user_input then we remove them from the training data
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
            pred = pred + trained_m.predict(user_input)

        pred = pred / len(models)

    return np.exp(pred)[0]


def score_models(models, load_models=False):
    """

     Prints the accuracy score for the provided models

     Parameters
     ----------
     models : list
             a list of model dictionaries containing the name of the filename
             and the function name

             Example: [ {'name': 'LinearRegression.sav','func': linear_reg},
                         {'name': 'Lasso.sav','func': lasso} ]


     load_models : boolean, optional, default False
                 can be overridden to True to score saved models

     Returns
     ----------
     None
     """

    train_data, train_labels, train_test_data_, test_data, test_labels = process_data()


    if load_models:
        models = load_models_from_file(models)


    for model in models:
        trained_m = model['func'](train_data, train_labels)
        pred = trained_m.predict(test_data)

        print(trained_m['name'])
        print("RMSLE ", mean_squared_log_error(test_labels, pred))
        print("RMSE ", mean_squared_error(test_labels, pred))
        print("Variance ", explained_variance_score(test_labels, pred))


def save_models_to_file(models):

    """
       Train the provided models and save them out

       Parameters
       ----------
       models : list
               a list of model dictionaries containing the name of the filename
               and the function name

               Example: [ {'name': 'LinearRegression.sav','func': linear_reg},
                           {'name': 'Lasso.sav','func': lasso} ]


       Returns
       ----------
       None

       """

    train_data, train_labels, train_test_data_, test_data, test_labels = process_data(save_train_data=True)

    for model in models:
        trained_m = model['func'](train_data, train_labels)

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        SPATH = os.path.join(BASE_DIR, 'predictModel/predictionModels/')
        SPATH = os.path.join(SPATH, model['name'])

        pickle.dump(trained_m, open(SPATH, 'wb'))
        print("Saved " + model['name'] + " to ", SPATH)


def load_models_from_file(models):

    """
          Load the requested models from file

          Parameters
          ----------
          models : list
                  a list of model dictionaries containing the name of the filename
                  and the function name

                  Example: [ {'name': 'LinearRegression.sav','func': linear_reg},
                              {'name': 'Lasso.sav','func': lasso} ]


          Returns
          ----------
          loaded_models : list

          """

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    loaded_models = []
    for model in models:
        SPATH = os.path.join(BASE_DIR, 'predictModel/predictionModels/')
        SPATH = os.path.join(SPATH, model['name'])
        loaded_models.append(pickle.load(open(SPATH, 'rb')))

    return loaded_models


def process_data(save_train_data=False):
    """
          Load the training data and process it

          Parameters
          ----------
          save_train_data : boolean, default True
                  Can be overridden to save out the [training_test_data, featureOrder] files after processing



          Returns
          ----------
          train_data : pandas dataframe
          train_labels : pandas dataframe
          train_test_data_ : pandas dataframe
          test_data : pandas dataframe
          test_labels : pandas dataframe


          """

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSVPATH = os.path.join(BASE_DIR,'predictModel/iowaHomes.csv')
    data = pd.read_csv(CSVPATH)

    data = data.drop(columns='Id')
    data = drop_outliers(data)


    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    train_labels = train_data['SalePrice'].apply(np.log)
    train_data = train_data.drop(columns='SalePrice')

    test_labels = test_data['SalePrice'].apply(np.log)
    test_data = test_data.drop(columns='SalePrice')

    train_data = fill_missing_values(train_data)
    test_data = fill_missing_values(test_data)

    train_data = feature_engineering(train_data)

    test_data = feature_engineering(test_data)

    train_test_data = pd.concat([train_data, test_data])  # data joined to perform factorization and scaling


    train_test_data_ = copy.deepcopy(train_test_data)

    if save_train_data:
        TDPATH = os.path.join(BASE_DIR, 'predictModel/predictionModels/training_data/training_data.csv')
        train_test_data.to_csv(TDPATH,index=False)  # saving out data with no missing values to use when making predictions in production

    train_test_data = encode_values(train_test_data)

    train_test_data = log_scale_values(train_test_data, train_test_data)

    train_test_data = drop_columns(train_test_data)

    test_data = train_test_data[len(train_data):]
    train_data = train_test_data[:len(train_data)]


    droppedCols = []
    while True:
        # using the stats package, this iteratively gets the feature with the highest pvalue
        # and drops it until there are no features left that are below the defined h threshold
        ols = linear_model.LinearRegression()
        ols = ols.fit(train_data, train_labels)

        xlabels = list(train_data.columns)
        statCoef = list(stats.coef_pval(ols, train_data,train_labels))

        h = 0
        for x in range(len(statCoef)-1):
            if h < statCoef[x+1]:
                # loop through all features to find the one with the highest pvalue
                h = statCoef[x+1]

        if h > 0.05:
            # if feature has a pvalue greater than h then drop it
            idx = statCoef.index(h) - 1
            droppedCols.append(xlabels[idx])
            train_data = train_data.drop(columns=xlabels[idx])
            test_data = test_data.drop(columns=xlabels[idx])

        else:
            break


    if save_train_data:
        # saves out data that is used to run an offline prediction
        stats.summary(ols, train_data, train_labels, xlabels=xlabels)
        TDPATH = os.path.join(BASE_DIR, 'predictModel/predictionModels/training_data/featureOrder.sav')

        featureOrder = []
        for x in range(len(statCoef) - 1):
            featureOrder.append(xlabels[x])
        pickle.dump(featureOrder, open(TDPATH, 'wb'))

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


dropped_features = ["Alley",
                   "Street", 'PoolQC', 'Utilities', 'RoofStyle',
                   'RoofMatl', "PoolArea", 'BsmtFinSF1', 'BsmtFinSF2', 'GarageQual',
                   'Exterior2nd',

                   'BsmtFinType2',
                   'YrSold', 'MoSold', 'SaleCondition', 'SaleType'  # these are dropped because we can't
                   # estimate the price of a house based on these as the house hasn't been sold yet

                   ]

for feat in engineered_features:
    #any features that are used to create new features are added to the list of dropped features
    for dep in feat['dependencies']:
        dropped_features.append(dep)


if __name__ == "__main__":
    pass

