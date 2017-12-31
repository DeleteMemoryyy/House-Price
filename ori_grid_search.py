# -*- coding: UTF-8 -*-
import time
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing as prep
from sklearn import ensemble
from sklearn import linear_model as lm
from sklearn import svm as svm
from sklearn import metrics as metrics
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None

    train_data = pd.read_csv('data/trainData.csv')
    test_data = pd.read_csv('data/testData.csv')
    all_data = pd.read_csv('data/allData.csv')
    proc_data = pd.concat([train_data, test_data])
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

    ori_to_numerical_columns = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                                'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC']

    ori_numerical_columns = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                            'TotalBsmtSF', 'Heating', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

    drop_columns = ['Alley', 'MasVnrType',
                    'MasVnrArea', 'Utilities', 'PoolQC', 'Fence']

    one_hot_columns = ['MSSubClass', 'MSZoning', 'LotShape', 'LandContour', 'Street', 'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Foundation', 'Heating', 'CentralAir',
                    'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'PavedDrive', 'MiscFeature', 'SaleType', 'SaleCondition']

    map_columns = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']

    numerical_columns = ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GrLivArea', 'BsmtFinSF1',
                        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

    other_columns = ['Id', 'SalePrice']

    # preprocess features
    # process LotFrontage
    TempLotFrontage = proc_data['LotFrontage']
    TempLotFrontage[proc_data['LotFrontage'].isnull()] = np.sqrt(
        proc_data['LotArea'][proc_data['LotFrontage'].isnull()])
    proc_data['LotFrontage'] = TempLotFrontage

    # process Condition
    Condition = pd.DataFrame(np.zeros((proc_data.shape[0], 9), dtype=int), columns=[
                            'Cond_Artery', 'Cond_Feedr', 'Cond_Norm', 'Cond_RRNn', 'Cond_RRAn', 'Cond_PosN', 'Cond_PosA', 'Cond_RRNe', 'Cond_RRAe'])
    for i in range(proc_data.shape[0]):
        col_name = "Cond_{}".format(proc_data['Condition1'][i])
        Condition.loc[i, col_name] = 1
        col_name = "Cond_{}".format(proc_data['Condition2'][i])
        Condition.loc[i, col_name] = 1
    proc_data = pd.concat((proc_data, Condition), axis=1).drop(
        ['Condition1', 'Condition2'], axis=1)

    # process Exterior
    Exterior = pd.DataFrame(np.zeros((proc_data.shape[0], 17), dtype=int), columns=[
                            'Ext_AsbShng', 'Ext_AsphShn', 'Ext_BrkComm', 'Ext_BrkFace', 'Ext_CBlock', 'Ext_CemntBd', 'Ext_HdBoard', 'Ext_ImStucc', 'Ext_MetalSd', 'Ext_Other', 'Ext_Plywood', 'Ext_PreCast', 'Ext_Stone', 'Ext_Stucco', 'Ext_VinylSd', 'Ext_Wd Sdng', 'Ext_WdShing'])
    proc_data.loc[proc_data['Exterior2nd'] == 'Brk Cmn', 'Exterior2nd'] = 'BrkComm'
    proc_data.loc[proc_data['Exterior2nd'] == 'CmentBd', 'Exterior2nd'] = 'CemntBd'
    proc_data.loc[proc_data['Exterior2nd'] == 'Wd Shng', 'Exterior2nd'] = 'Wd Sdng'
    for i in range(proc_data.shape[0]):
        col_name = "Ext_{}".format(proc_data['Exterior1st'][i])
        if col_name not in Exterior.columns:
            Exterior.loc[i, col_name] = 1
        col_name = "Ext_{}".format(proc_data['Exterior2nd'][i])
        if col_name not in Exterior.columns:
            Exterior.loc[i, col_name] = 1
    proc_data = pd.concat((proc_data, Exterior), axis=1).drop(
        ['Exterior1st', 'Exterior2nd'], axis=1)

    # process basement
    proc_data['BsmtCond'] = proc_data['BsmtCond'].fillna('TA')
    proc_data['BsmtExposure'] = proc_data['BsmtExposure'].fillna('No')
    proc_data['BsmtFinType1'] = proc_data['BsmtFinType1'].fillna('Unf')
    proc_data['BsmtFinType2'] = proc_data['BsmtFinType2'].fillna('Unf')
    proc_data['BsmtQual'] = proc_data['BsmtQual'].fillna('TA')
    proc_data['BsmtExposure'] = proc_data['BsmtExposure'].replace(
        {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0})
    proc_data['BsmtFinType1'] = proc_data['BsmtFinType1'].replace(
        {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0})
    proc_data['BsmtFinType2'] = proc_data['BsmtFinType2'].replace(
        {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0})

    # process Electrical
    proc_data['Electrical'] = proc_data['Electrical'].fillna('SBrkr')

    # process FireplaceQu
    proc_data['FireplaceQu'] = proc_data['FireplaceQu'].fillna('None')

    # process garage
    proc_data['GarageType'] = proc_data['GarageType'].fillna('None')
    proc_data['GarageYrBlt'] = proc_data['GarageYrBlt'].fillna(
        proc_data['GarageYrBlt'].mean())
    proc_data['GarageFinish'] = proc_data['GarageFinish'].fillna('None')
    proc_data['GarageCars'] = proc_data['GarageCars'].fillna(
        proc_data['GarageCars'].mean())
    proc_data['GarageArea'] = proc_data['GarageArea'].fillna(
        proc_data['GarageArea'].mean())
    proc_data['GarageQual'] = proc_data['GarageQual'].fillna('TA')
    proc_data['GarageCond'] = proc_data['GarageCond'].fillna('TA')

    # process MiscFeature
    proc_data['MiscFeature'] = proc_data['MiscFeature'].fillna('NA')

    # map encode
    for col in map_columns:
        proc_data[col] = proc_data[col].replace(
            {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': '0'})

    # one-hot processing
    proc_data = pd.get_dummies(proc_data, columns=one_hot_columns)

    # drop features
    proc_data = proc_data.drop(drop_columns, axis=1)

    # standardization
    ss_x = prep.StandardScaler()
    proc_data[numerical_columns] = ss_x.fit_transform(
        proc_data[numerical_columns].values)

    # spilt training data
    x_all_train = proc_data[proc_data['SalePrice']
                            != 0].drop(other_columns, axis=1)
    x_test = proc_data[proc_data['SalePrice'] == 0].drop(other_columns, axis=1)
    y_all_train = proc_data['SalePrice'][proc_data['SalePrice'] != 0].values
    y_test = all_data['SalePrice'][proc_data['SalePrice'] == 0].values

    #%% SVR grid search
    svr_grid = GridSearchCV(svm.SVR(), param_grid={'C':[400,500,600], 'gamma': [0.0001,0.001]}, cv=4,
                        scoring='neg_mean_squared_error',verbose=1,n_jobs=4)
    svr_grid.fit(x_all_train,y_all_train)
    print('Best svr parameters: {}'.format(svr_grid.best_params_))

    #%% SVR
    svr = svr_grid.best_estimator_
    svr_score = -cross_val_score(svr,x_all_train,y_all_train,cv=4,scoring='neg_mean_squared_error')
    svr.fit(x_all_train, y_all_train)
    svr_y_all_predict = svr.predict(x_all_train)
    svr_y_test_predict = svr.predict(x_test)
    print("svr_valid_rmse: {}".format(np.sqrt(svr_score.mean())))
    print("svr_all_rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_all_train,svr_y_all_predict))))

    # #%% GradientBoosting regression
    # gbr_grid = GridSearchCV(ensemble.GradientBoostingRegressor(loss='huber',max_features='sqrt'),param_grid={
    #     'n_estimators': [25, 50, 100, 200], 'learning_rate':[0.001,0.01,0.1],'max_depth':[3,5,7]},scoring='neg_mean_squared_error',verbose=1,n_jobs=4)
    # gbr_grid.fit(x_all_train,y_all_train)
    # print('Best gbr parameters: {}'.format(gbr_grid.best_params_))

    # #%% GBR
    # gbr = gbr_grid.best_estimator_
    # gbr_score = -cross_val_score(gbr, x_all_train,
    #                              y_all_train, cv=4, scoring='neg_mean_squared_error')
    # gbr.fit(x_all_train,y_all_train)
    # gbr_y_all_predict = gbr.predict(x_all_train)
    # gbr_y_test_predict = gbr.predict(x_test)
    # print("gbr_valid_mse: {}".format(gbr_score.mean()))
    # print("gbr_all_mse: {}".format(
    #     metrics.mean_squared_error(y_all_train, gbr_y_all_predict)))

    # #%% RandomForest regression
    # rfr_grid = GridSearchCV(ensemble.RandomForestRegressor(), param_grid={'n_estimators': range(10,111,20), 'max_depth': [3, 5, 7, None]}, scoring='neg_mean_squared_error', verbose=1, n_jobs=4)
    # rfr_grid.fit(x_all_train,y_all_train)
    # print('Best grfr parameters: {}'.format(rfr_grid.best_params_))

    # #%% GBR
    # rfr = rfr_grid.best_estimator_
    # rfr_score = -cross_val_score(rfr, x_all_train,
    #                              y_all_train, cv=4, scoring='neg_mean_squared_error')
    # rfr.fit(x_all_train, y_all_train)
    # rfr_y_all_predict = rfr.predict(x_all_train)
    # rfr_y_test_predict = rfr.predict(x_test)
    # print("rfr_valid_mse: {}".format(rfr_score.mean()))
    # print("rfr_all_mse: {}".format(
    #     metrics.mean_squared_error(y_all_train, rfr_y_all_predict)))

    # #%% RidgeCV search
    # regr_cv = lm.RidgeCV(alphas=[0.1, 1, 11, 21],
    #                      scoring='neg_mean_squared_error')
    # regr_cv.fit(x_all_train, y_all_train)
    # print('Best Ridge alpha: {}'.format(regr_cv.alpha_))

    # #%% Ridge regression
    # regr = lm.Ridge(alpha=regr_cv.alpha_)
    # regr_score = -cross_val_score(
    #     regr, x_all_train, y_all_train, cv=4, scoring='neg_mean_squared_error')
    # regr.fit(x_all_train, y_all_train)
    # regr_y_all_predict = regr.predict(x_all_train)
    # regr_y_test_predict = regr.predict(x_test)
    # print("regr_valid_mse: {}".format(regr_score.mean()))
    # print("regr_all_mse: {}".format(
    #     metrics.mean_squared_error(y_all_train, regr_y_all_predict)))

    # #%% LassoCV search
    # lsr_cv = lm.LassoCV()
    # lsr_cv.fit(x_all_train, y_all_train)
    # print('Best Lasso alpha: {}'.format(lsr_cv.alpha_))

    # #%% Lasso regression
    # lsr = lm.Ridge(alpha=lsr_cv.alpha_)
    # lsr_score = -cross_val_score(
    #     lsr, x_all_train, y_all_train, cv=4, scoring='neg_mean_squared_error')
    # lsr.fit(x_all_train, y_all_train)
    # lsr_y_all_predict = lsr.predict(x_all_train)
    # lsr_y_test_predict = lsr.predict(x_test)
    # print("lsr_valid_mse: {}".format(lsr_score.mean()))
    # print("lsr_all_mse: {}".format(
    #     metrics.mean_squared_error(y_all_train, lsr_y_all_predict)))

    # #%% ElasticNetCV search
    # enr_cv = lm.ElasticNetCV(l1_ratio=0.5)
    # enr_cv.fit(x_all_train, y_all_train)
    # print('Best ElasticNet alpha: {0}, l1_ratio: {1}'.format(
    #     enr_cv.alpha_, enr_cv.l1_ratio_))

    # #%% Elastic Net regression
    # enr = lm.ElasticNet(alpha=enr_cv.alpha_, l1_ratio=enr_cv.l1_ratio_)
    # enr_score = -cross_val_score(
    #     enr, x_all_train, y_all_train, cv=4, scoring='neg_mean_squared_error')
    # enr.fit(x_all_train, y_all_train)
    # enr_y_all_predict = enr.predict(x_all_train)
    # enr_y_test_predict = enr.predict(x_test)
    # print("enr_valid_mse: {}".format(enr_score.mean()))
    # print("enr_all_mse: {}".format(
    #     metrics.mean_squared_error(y_all_train, enr_y_all_predict)))
