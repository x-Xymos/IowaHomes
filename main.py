import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def replace_null(x):
    if pd.isnull(x):
        return "NA"
    else:
        return x

def factorize_data(d):
    for x in range(d.shape[1]):
        if d.iloc[:, x].dtype == "object":
            d.iloc[:, x] = pd.factorize(d.iloc[:, x])[0]

    return d

data = pd.read_csv('iowaHomes.csv')

#replacing null categorical values
data['FireplaceQu'] = data['FireplaceQu'].map(replace_null)
data['BsmtQual'] = data['BsmtQual'].map(replace_null)
data['BsmtCond'] = data['BsmtCond'].map(replace_null)
data['BsmtExposure'] = data['BsmtExposure'].map(replace_null)
data['BsmtFinType1'] = data['BsmtFinType1'].map(replace_null)
data['BsmtFinType2'] = data['BsmtFinType2'].map(replace_null)
data['GarageType'] = data['GarageType'].map(replace_null)
data['MasVnrType'] = data['MasVnrType'].map(lambda x: "None" if pd.isnull(x) else x)
data['Electrical'] = data['Electrical'].map(lambda x: "SBrkr" if pd.isnull(x) else x)
data['GarageQual'] = data['GarageQual'].map(replace_null)
data['GarageCond'] = data['GarageCond'].map(replace_null)
data['GarageFinish'] = data['GarageFinish'].map(replace_null)
data['GarageType'] = data['GarageType'].map(replace_null)
data['PoolQC'] = data['PoolQC'].map(replace_null)
data['Fence'] = data['Fence'].map(replace_null)
data['MiscFeature'] = data['MiscFeature'].map(replace_null)

#replacing null numeric values
data['MasVnrArea'] = data['MasVnrArea'].map(lambda x: 0 if pd.isnull(x) else x)


#print(data.isnull().sum())
#print(data['MasVnrArea'].value_counts())

data = data.drop(columns=["Id", "Alley",
                          "PoolArea",
                          "MoSold",
                          "3SsnPorch",
                          "BsmtFinSF2",
                          "BsmtHalfBath",
                          "MiscVal",
                          "LowQualFinSF",
                          "YrSold",
                          "OverallCond",
                          "ScreenPorch"])
#dropping a lot of attributes with low correlation to reduce noise
#seems to have improved the variance score a little bit

#corr_matrix = data.corr()
#print(corr_matrix["SalePrice"].sort_values(ascending=False))


#trying out different data scaling functions
distributions = [
    ('Unscaled data', "NA"),
    ('Data after min-max scaling',
        MinMaxScaler().fit_transform),
    ('Data after max-abs scaling',
        MaxAbsScaler().fit_transform),
    ('Data after robust scaling',
        RobustScaler(quantile_range=(25, 75)).fit_transform),
    ('Data after power transformation (Yeo-Johnson)',
     PowerTransformer(method='yeo-johnson').fit_transform),
    ('Data after power transformation (Box-Cox)',
     PowerTransformer().fit_transform),
    ('Data after quantile transformation (uniform pdf)',
        QuantileTransformer(output_distribution='uniform')
        .fit_transform),
    ('Data after sample-wise L2 normalizing',
        Normalizer().fit_transform),
    ('Data after standardization',
     StandardScaler().fit_transform)
]

for dist in distributions:
    cleanData = data.copy()


    dataNorm = cleanData.drop(columns=["SalePrice",
                                  "GarageYrBlt", "YearRemodAdd",
                                  "YearBuilt", "MSSubClass", "OverallQual",
                                  "BsmtFullBath", 'FullBath', 'HalfBath',
                                  "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
                                  "Fireplaces", "GarageCars"
                                  ])
    dataNorm = dataNorm.select_dtypes(include=[np.number])
    dataNorm_col = dataNorm.select_dtypes(include=[np.number]).columns


    if dist[1] != "NA":
        try:
            dataNorm = dist[1](dataNorm)
            cleanData[dataNorm_col] = dataNorm
        except Exception as exc:
            print(exc)
            continue

    cleanData = factorize_data(cleanData)

    from sklearn.model_selection import KFold

    scores = []

    kf = KFold(n_splits=10, random_state=42)
    for train, test in kf.split(cleanData):

        train_data = cleanData.drop(columns='SalePrice').iloc[train]
        train_labels = cleanData['SalePrice'].iloc[train]

        test_data = cleanData.drop(columns='SalePrice').iloc[test]
        test_labels = cleanData['SalePrice'].iloc[test]

        imputer = SimpleImputer(missing_values=np.nan, strategy='median')

        train_data[['LotFrontage']] = imputer.fit_transform(train_data[['LotFrontage']])
        train_data[['GarageYrBlt']] = imputer.fit_transform(train_data[['GarageYrBlt']])

        test_data[['LotFrontage']] = imputer.fit_transform(test_data[['LotFrontage']])
        test_data[['GarageYrBlt']] = imputer.fit_transform(test_data[['GarageYrBlt']])

        #linear regression
        # reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1,
        #                  normalize=False)
        # reg.fit(train_data, train_labels)
        # pred = reg.predict(test_data)
        # print(pred[1],test_labels.iloc[1])
        # acc = explained_variance_score(test_labels, pred)
        # scores.append(acc)
        ##########################################################################

        #random forest regressor

        from sklearn.model_selection import GridSearchCV

        param_grid = [
            {'n_estimators': [50, 100, 150, 200], 'max_features': [4, 8, 32, 64], 'n_jobs':[-1]},
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]


        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(n_estimators=100, max_features=32, random_state=42, n_jobs=-1)

        #grid_search = GridSearchCV(reg, param_grid, cv=5,scoring='r2')

        reg.fit(train_data, train_labels)
        #print(grid_search.best_estimator_)
        pred = reg.predict(test_data)
        #print(pred[3],test_labels.iloc[3])
        scores.append(explained_variance_score(test_labels, pred))
        ##########################################################################

        # #############################################################################
        # Lasso
        # from sklearn.linear_model import Lasso
        #
        # alpha = 0.1
        # lasso = Lasso(alpha=alpha,max_iter=5000, random_state=42)
        #
        # y_pred_lasso = lasso.fit(train_data, train_labels).predict(test_data)
        # r2_score_lasso = r2_score(test_labels, y_pred_lasso)
        # scores.append(r2_score_lasso)

        # #############################################################################


        #lars
        # from sklearn import linear_model
        #
        # reg = linear_model.Lars(n_nonzero_coefs=25)
        # y_pred = reg.fit(train_data, train_labels).predict(test_data)
        # r2_score_sgd = r2_score(test_labels, y_pred)
        # scores.append(r2_score_sgd)
        #############################3

    print(dist[0])
    print(np.mean(scores))















