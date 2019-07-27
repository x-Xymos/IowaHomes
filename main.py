import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv('iowaHomes.csv')

#according to the info.txt file NA = NO for most categories, for example no fireplace,
#no basement, does that mean missing values = NO???

def replace_null(x):
    if pd.isnull(x):
        return "NA"
    else:
        return x


#replacing null categorical values
data['FireplaceQu'] = data['FireplaceQu'].map(replace_null)
data['BsmtQual'] = data['BsmtQual'].map(replace_null)
data['BsmtCond'] = data['BsmtCond'].map(replace_null)
data['BsmtExposure'] = data['BsmtExposure'].map(replace_null)
data['BsmtFinType1'] = data['BsmtFinType1'].map(replace_null)
data['BsmtFinType2'] = data['BsmtFinType2'].map(replace_null)
data['GarageType'] = data['GarageType'].map(replace_null)
#??? combine basement attributes into basement or no basement ???
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
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
data[['LotFrontage']] = imputer.fit_transform(data[['LotFrontage']])
data['MasVnrArea'] = data['MasVnrArea'].map(lambda x: 0 if pd.isnull(x) else x)
data[['GarageYrBlt']] = imputer.fit_transform(data[['GarageYrBlt']])

#??? combine garage attributes into garage or no garage ???


#print(data.isnull().sum())
#print(data['Alley'].value_counts())

#todo compare model performance
# with and without replaced data

data = data.drop(columns=["Id", "Alley"])

corr_matrix = data.corr()
#print(corr_matrix["SalePrice"].sort_values(ascending=False))

def factorize_data(d):
    for x in range(d.shape[1]):
        if d.iloc[:, x].dtype == "object":
            d.iloc[:, x] = pd.factorize(d.iloc[:, x])[0]

    return data

data['MSSubClass'] = pd.factorize(data['MSSubClass'])[0]

dataNorm = data.drop(columns=["SalePrice", "YrSold", "MoSold",
                              "GarageYrBlt", "YearRemodAdd",
                              "YearBuilt", "MSSubClass"
                              ])


dataNorm = dataNorm.select_dtypes(include=[np.number])
dataNorm_col = dataNorm.select_dtypes(include=[np.number]).columns

#trying out different data scaling functions
distributions = [
    ('Unscaled data', dataNorm),
    ('Data after standard scaling',
        StandardScaler().fit_transform(dataNorm)),
    ('Data after min-max scaling',
        MinMaxScaler().fit_transform(dataNorm)),
    ('Data after max-abs scaling',
        MaxAbsScaler().fit_transform(dataNorm)),
    ('Data after robust scaling',
        RobustScaler(quantile_range=(25, 75)).fit_transform(dataNorm)),
    ('Data after power transformation (Yeo-Johnson)',
     PowerTransformer(method='yeo-johnson').fit_transform(dataNorm)),
    ('Data after power transformation (Box-Cox)',
     PowerTransformer().fit_transform(dataNorm)),
    ('Data after quantile transformation (gaussian pdf)',
        QuantileTransformer(output_distribution='normal')
        .fit_transform(dataNorm)),
    ('Data after quantile transformation (uniform pdf)',
        QuantileTransformer(output_distribution='uniform')
        .fit_transform(dataNorm)),
    ('Data after sample-wise L2 normalizing',
        Normalizer().fit_transform(dataNorm)),
]

for dist in distributions:
    print(dist[0])
    dataNorm = dist[1]

#dataNorm = preprocessing.QuantileTransformer(output_distribution="normal").fit_transform(dataNorm)

    data[dataNorm_col] = dataNorm

    data = factorize_data(data)

    #print(data)


    y = data['SalePrice']
    x = data.drop(columns='SalePrice')


    train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.3, random_state=42)

    reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1,
                     normalize=False)
    reg.fit(train_data, train_labels)

    pred = reg.predict(test_data)
    acc = explained_variance_score(test_labels, pred)

    print(acc)







#print(data.info())
# pca = PCA().fit(data)
# print('Explained variance by component: %s' % pca.explained_variance_ratio_)
# print(pd.DataFrame(pca.components_, columns=data.feature_names))

#print(data.info())

#data.iloc[:, 1:]

#y = data.iloc[:, 0]
#x = data.drop(columns=1)











