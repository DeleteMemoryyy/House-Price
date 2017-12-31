import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model as lm
from sklearn import kernel_ridge
from sklearn import svm as svm
from sklearn import metrics
import rgf.sklearn as rgf
import xgboost as xgb
import lightgbm as lgb

from dataloader import DataLoader, DataConfig
# from stastical_regression import grid_search, varify_on_test

dl = DataLoader(DataConfig())

lsr = lm.Lasso(alpha=0.00006)
regr = lm.Ridge(alpha=0.01)
enr = lm.ElasticNet(alpha=0.0001, l1_ratio=0.5)
# krr = kernel_ridge.KernelRidge(kernel='polynomial')
krr = kernel_ridge.KernelRidge(kernel='linear',alpha=0.01,gamma=0.0001)
svr = svm.SVR(C=15, gamma=0.001)
gbr = ensemble.GradientBoostingRegressor(
    loss='huber', max_features='sqrt', n_estimators=400,learning_rate=0.1,max_depth=3)
rfr = ensemble.RandomForestRegressor(n_estimators=200,max_depth=9)
xgbr = xgb.XGBRegressor(booster='gbtree', gamma=0.001,
                        max_depth=11, min_child_weight=1,n_estimators=150)
xgblr = xgb.XGBRegressor(booster='gblinear', n_estimators=4000, gamma=0.0001,learning_rate=0.35,max_depth=13)
lgbr = lgb.LGBMRegressor(num_leaves=7, min_data_in_leaf=7,
                        learning_rate=0.05, n_estimators=300)
rgfr = rgf.RGFRegressor(max_leaf=700, learning_rate=0.1,
                        min_samples_leaf=1, test_interval=10)


#%%
models = [rgfr]
model_names = ['rgfr']
ft = {33: 0.15856, 169: -0.13334}
param_grid = [{'max_leaf': [700], 'test_interval':[10],
               'min_samples_leaf':[1], 'learning_rate':[0.1]}]
# for idx, model in enumerate(models):
#     best_model = grid_search(model, dl, param_grid[idx],cv=2,verbose=0,model_name=model_names[idx])
#     train_rmse, test_rmse = varify_on_test(best_model, dl)
#     print('{0:s}  train_rmse: {1:f}'.format(
#         model_names[idx], train_rmse))
#     print('{0:s}  test_rmse: {1:f}'.format(model_names[idx], test_rmse))
