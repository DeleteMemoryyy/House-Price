import time
import warnings
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
from stastical_regression import Stacking, varify_on_test

warnings.filterwarnings('ignore')

save_file = True
enable_stacking = True
enable_add_result = False
valide_all = False
valide_on_test = True

lsr = lm.Lasso(alpha=0.00006)
regr = lm.Ridge(alpha=0.01)
enr = lm.ElasticNet(alpha=0.0001, l1_ratio=0.5)
# krr = kernel_ridge.KernelRidge(kernel='polynomial')
krr = kernel_ridge.KernelRidge(kernel='linear', alpha=0.01, gamma=0.0001)
svr = svm.SVR(C=15, gamma=0.001)
gbr = ensemble.GradientBoostingRegressor(
    loss='huber', max_features='sqrt', n_estimators=400, learning_rate=0.1, max_depth=3)
rfr = ensemble.RandomForestRegressor(n_estimators=200, max_depth=9)
xgbr = xgb.XGBRegressor(booster='gbtree', gamma=0.001,
                        max_depth=11, min_child_weight=1, n_estimators=150)
xgblr = xgb.XGBRegressor(booster='gblinear', n_estimators=4000,
                         gamma=0.0001, learning_rate=0.35, max_depth=13)
lgbr = lgb.LGBMRegressor(num_leaves=7, min_data_in_leaf=7,
                         learning_rate=0.05, n_estimators=300)
rgfr = rgf.RGFRegressor(max_leaf=700, learning_rate=0.1,
                        min_samples_leaf=1, test_interval=10)

dl = DataLoader(DataConfig())

#%%
if enable_stacking:
    # base_models = [lsr,regr,enr,krr,gbr,xgbr]
    # base_model_names = ['lsr','regr','engr','krr','gbr','xgbr']
    base_models = [lsr,regr,enr,krr,gbr]
    base_model_names = ['lsr','regr','engr','krr','gbr']
    stacker = [lm.Ridge(alpha=0.0065)]
    added_result = []
    weights = [1.0]

    if valide_all:
        for idx in range(len(base_models)):
            train_rmse, test_rmse = varify_on_test(base_models[idx], dl,model_name=base_model_names[idx])
            print('{0:s}  train_rmse: {1:f}'.format(
                base_model_names[idx], train_rmse))
            print('{0:s}  test_rmse: {1:f}'.format(base_model_names[idx], test_rmse))

    stacking = Stacking(5,base_models,stacker,added_result,weights)
    stacking_y_test_predict = dl.ss_y.inverse_transform(stacking.fit_predict(dl.x_train,dl.y_prep_train,dl.x_test).reshape((-1,1)))

    # stacking.fit(dl.x_train,dl.y_prep_train)
    # stacking_y_test_predict = dl.ss_y.inverse_transform(stacking.predict(dl.x_test).reshape((-1,1)))

    if valide_on_test:
        test_rmse = np.sqrt(metrics.mean_squared_error(
            dl.y_test, stacking_y_test_predict))
        print('Stacking_test_rmse: {}'.format(test_rmse))

    if save_file:
        result_data = np.array(
            [[i for i in range(1, stacking_y_test_predict.shape[0] + 1)], stacking_y_test_predict[:,0]]).transpose()
        result = pd.DataFrame(result_data, columns=['Id', 'SalePrice'])
        result['Id'] = result['Id'].astype('int')
        save_name='result/result_{0:s}_stacking.csv'.format(time.strftime('%b_%d_%H-%M-%S', time.localtime()))
        result.to_csv(save_name, header=True,
                    index=False, encoding='utf-8')
        print('save to {}'.format(save_name))
