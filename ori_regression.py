# -*- coding: UTF-8 -*-
#%% init
import time
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import preprocessing as prep
from sklearn import ensemble
from sklearn import linear_model as lm
from sklearn import svm
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns

train_data = pd.read_csv('data/trainData.csv')
test_data = pd.read_csv('data/testData.csv')
all_data = pd.read_csv('data/allData.csv')
proc_data = pd.concat([train_data,test_data])
proc_data.index = range(proc_data.shape[0])
proc_data['SalePrice'] = proc_data['SalePrice'].fillna(0)

svr_C = 200
svr_gamma = 0.001
regr_alpha = 11.0
lsr_alpha = 0.0005547
enr_alpha = 0.0009649
enr_l1r = 0.5
gbr_n_estimators = 400
rfr_n_estimators = 90

rand_seed = 2017

ori_one_hot_columns = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                       'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

ori_one_condition_columns = [
    'Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd', 'Foundation']

ori_to_numerical_columns = ['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','KitchenQual','FireplaceQu','GarageFinish','GarageQual','GarageCond','PoolQC']

ori_numerical_columns = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                         'TotalBsmtSF', 'Heating', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']

drop_columns = ['Alley', 'MasVnrType',
                'MasVnrArea', 'Utilities', 'PoolQC', 'Fence']

one_hot_columns = ['MSSubClass', 'MSZoning', 'LotShape', 'LandContour', 'Street', 'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Foundation', 'Heating', 'CentralAir',
                   'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'PavedDrive', 'MiscFeature', 'SaleType', 'SaleCondition']

map_columns = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC','KitchenQual','GarageQual','GarageCond']

numerical_columns = ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GrLivArea', 'BsmtFinSF1',
                     'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

other_columns = ['Id', 'SalePrice']

# preprocess features
# process LotFrontage
TempLotFrontage = proc_data['LotFrontage']
TempLotFrontage[proc_data['LotFrontage'].isnull()] = np.sqrt(proc_data['LotArea'][proc_data['LotFrontage'].isnull()])
proc_data['LotFrontage'] = TempLotFrontage

# process Condition
Condition = pd.DataFrame(np.zeros((proc_data.shape[0],9),dtype=int),columns=['Cond_Artery','Cond_Feedr','Cond_Norm','Cond_RRNn','Cond_RRAn','Cond_PosN','Cond_PosA','Cond_RRNe','Cond_RRAe'])
for i in range(proc_data.shape[0]):
    col_name = "Cond_{}".format(proc_data['Condition1'][i])
    Condition.loc[i,col_name] = 1
    col_name = "Cond_{}".format(proc_data['Condition2'][i])
    Condition.loc[i,col_name] = 1
proc_data = pd.concat((proc_data,Condition),axis=1).drop(['Condition1','Condition2'],axis=1)

# process Exterior
Exterior = pd.DataFrame(np.zeros((proc_data.shape[0], 17), dtype=int), columns=[
                        'Ext_AsbShng', 'Ext_AsphShn', 'Ext_BrkComm', 'Ext_BrkFace', 'Ext_CBlock', 'Ext_CemntBd', 'Ext_HdBoard', 'Ext_ImStucc', 'Ext_MetalSd', 'Ext_Other', 'Ext_Plywood', 'Ext_PreCast', 'Ext_Stone', 'Ext_Stucco', 'Ext_VinylSd', 'Ext_Wd Sdng', 'Ext_WdShing'])
proc_data.loc[proc_data['Exterior2nd'] == 'Brk Cmn', 'Exterior2nd'] = 'BrkComm'
proc_data.loc[proc_data['Exterior2nd'] == 'CmentBd', 'Exterior2nd'] = 'CemntBd'
proc_data.loc[proc_data['Exterior2nd'] == 'Wd Shng', 'Exterior2nd'] = 'Wd Sdng'
for i in range(proc_data.shape[0]):
    col_name = "Ext_{}".format(proc_data['Exterior1st'][i])
    if col_name not in Exterior.columns:
        Exterior.loc[i,col_name] = 1
    col_name = "Ext_{}".format(proc_data['Exterior2nd'][i])
    if col_name not in Exterior.columns:
        Exterior.loc[i, col_name] = 1
proc_data = pd.concat((proc_data,Exterior),axis=1).drop(['Exterior1st','Exterior2nd'],axis=1)

# process basement
proc_data['BsmtCond'] = proc_data['BsmtCond'].fillna('TA')
proc_data['BsmtExposure'] = proc_data['BsmtExposure'].fillna('No')
proc_data['BsmtFinType1'] = proc_data['BsmtFinType1'].fillna('Unf')
proc_data['BsmtFinType2'] = proc_data['BsmtFinType2'].fillna('Unf')
proc_data['BsmtQual'] = proc_data['BsmtQual'].fillna('TA')
proc_data['BsmtExposure'] = proc_data['BsmtExposure'].replace({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0})
proc_data['BsmtFinType1'] = proc_data['BsmtFinType1'].replace({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0})
proc_data['BsmtFinType2'] = proc_data['BsmtFinType2'].replace({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0})

# process Electrical
proc_data['Electrical'] = proc_data['Electrical'].fillna('SBrkr')

# process FireplaceQu
proc_data['FireplaceQu'] = proc_data['FireplaceQu'].fillna('None')

# process garage
proc_data['GarageType'] = proc_data['GarageType'].fillna('None')
proc_data['GarageYrBlt'] = proc_data['GarageYrBlt'].fillna(proc_data['GarageYrBlt'].mean())
proc_data['GarageFinish'] = proc_data['GarageFinish'].fillna('None')
proc_data['GarageCars'] = proc_data['GarageCars'].fillna(proc_data['GarageCars'].mean())
proc_data['GarageArea'] = proc_data['GarageArea'].fillna(proc_data['GarageArea'].mean())
proc_data['GarageQual'] = proc_data['GarageQual'].fillna('TA')
proc_data['GarageCond'] = proc_data['GarageCond'].fillna('TA')

# process MiscFeature
proc_data['MiscFeature'] = proc_data['MiscFeature'].fillna('NA')

# map encode
for col in map_columns:
    proc_data[col] = proc_data[col].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':'0'})

# one-hot processing
proc_data = pd.get_dummies(proc_data, columns=one_hot_columns)

# drop features
proc_data = proc_data.drop(drop_columns,axis=1)

# standardization
ss_x = prep.StandardScaler()
proc_data[numerical_columns] = ss_x.fit_transform(proc_data[numerical_columns].values)

# spilt training data
x_all_train = proc_data[proc_data['SalePrice'] != 0].drop(other_columns,axis=1)
x_test = proc_data[proc_data['SalePrice'] == 0].drop(other_columns, axis=1)
y_all_train = proc_data['SalePrice'][proc_data['SalePrice'] != 0].values
y_test = all_data['SalePrice'][proc_data['SalePrice'] == 0].values

#%% regression
svr = svm.SVR(C=svr_C, gamma=svr_gamma)
regr = lm.Ridge(alpha=regr_alpha)
lsr = lm.Lasso(alpha=lsr_alpha)
enr = lm.ElasticNet(alpha=lsr_alpha,l1_ratio=enr_l1r)
gbr = ensemble.GradientBoostingRegressor(loss='huber', max_features='sqrt',n_estimators=gbr_n_estimators)
rfr = ensemble.RandomForestRegressor(n_estimators=rfr_n_estimators)
class Stacking(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rand_seed)

        s_train = np.zeros((X.shape[0], len(self.base_models)))

        for i, mod in enumerate(self.base_models):
            j = 0
            for idx_train, idx_valid in kf.split(range(len(y))):
                x_train_j = X[idx_train]
                y_train_j = y[idx_train]
                x_valid_j = X[idx_valid]

                mod.fit(x_train_j, y_train_j)

                y_valid_j = mod.predict(x_valid_j)[:]
                s_train[idx_valid, i] = y_valid_j

                j += 1

        self.stacker.fit(s_train, y)

    def predict(self,T):
        T = np.array(T)
        s_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, mod in enumerate(self.base_models):
            s_test[:, i] = mod.predict(T)[:]

        y_predict = self.stacker.predict(s_test)[:]

        return y_predict

#%% 5-fold stacking
stacking = Stacking(n_folds=5, stacker=lm.Ridge(
    alpha=11.0), base_models=[enr, regr, lsr, svr, gbr, rfr])
folds = KFold(n_splits=5, shuffle=True, random_state=rand_seed).split(range(x_all_train.shape[0]))
stacking_score = []
for idx_train, idx_valid in folds:
    X = np.array(x_all_train)
    y = np.array(y_all_train)
    x_train = X[idx_train]
    x_valid = X[idx_valid]
    y_train = y[idx_train]
    y_valid = y[idx_valid]
    stacking.fit(x_train,y_train)
    stacking_y_valid_predict = stacking.predict(x_valid)
    stacking_score.append(metrics.mean_squared_error(y_valid, stacking_y_valid_predict))
stacking_score = np.array(stacking_score)
print('stacking_valid_mse: {}'.format(stacking_score.mean()))
print('stacking_valid_mse_std: {}'.format(stacking_score.std()))
stacking.fit(x_all_train, y_all_train)
stacking_y_all_predict = stacking.predict(x_all_train)
print('stacking_all_mse: {}'.format(
    metrics.mean_squared_error(y_all_train, stacking_y_all_predict)))
stacking_y_test_predict = stacking.predict(x_test)
