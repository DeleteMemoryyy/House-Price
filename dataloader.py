import numpy as np
import pandas as pd
from sklearn import preprocessing as prep

pd.options.mode.chained_assignment = None

class DataConfig(object):
    def __init__(self):
        self.train_file = 'data/trainData.csv'
        self.test_file = 'data/testData.csv'
        self.test_result_file = 'data/testResult.csv'
        self.all_file = 'data/allData.csv'

        self.ori_one_hot_columns = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                            'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

        self.ori_one_condition_columns = [
            'Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd', 'Foundation']

        self.ori_to_numerical_columns = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                                    'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC']

        self.ori_numerical_columns = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                                'TotalBsmtSF', 'Heating', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

        self.drop_columns = ['Alley', 'MasVnrType',
                        'MasVnrArea', 'Utilities', 'PoolQC', 'Fence']
        self.one_hot_columns = ['MSSubClass', 'MSZoning', 'LotShape', 'LandContour', 'Street', 'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Foundation', 'Heating', 'CentralAir',
                        'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'PavedDrive', 'MiscFeature', 'SaleType', 'SaleCondition']
        self.map_columns = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                    'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']
        self.numerical_columns = ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GrLivArea', 'BsmtFinSF1',
                            'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
        self.other_columns = ['Id', 'SalePrice', 'TestLabel']

class LogScaler(object):
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return np.log(X)

    def fit_transform(self, X):
        return np.log(X)

    def inverse_transform(self, X):
        return np.exp(X)


class DataLoader(object):
    def __init__(self, conf):
        self.train_data = pd.read_csv(conf.train_file)
        self.test_data = pd.read_csv(conf.test_file)
        self.test_result = pd.read_csv(conf.test_result_file)
        self.all_data = pd.read_csv(conf.all_file)
        proc_data = pd.concat([self.train_data, self.test_data],ignore_index=True)
        proc_data['SalePrice'] = proc_data['SalePrice'].fillna(0)
        proc_data['TestLabel'] = proc_data['SalePrice'] == 0

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
        for col in conf.map_columns:
            proc_data[col] = proc_data[col].replace(
                {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': '0'})

        # one-hot processing
        proc_data = pd.get_dummies(proc_data, columns=conf.one_hot_columns)

        # drop features
        proc_data = proc_data.drop(conf.drop_columns, axis=1)

        # standardization
        ss_x = prep.StandardScaler()
        proc_data[conf.numerical_columns] = ss_x.fit_transform(
            proc_data[conf.numerical_columns].values)

        # spilt training data
        self.x_train = proc_data[proc_data['TestLabel'] == False].drop(conf.other_columns, axis=1)
        self.x_test = proc_data[proc_data['TestLabel']].drop(conf.other_columns, axis=1)
        self.y_train = self.train_data['SalePrice'].values
        self.y_test = self.test_result['SalePrice'].values

        self.ss_y = LogScaler()
        # self.ss_y = prep.StandardScaler()
        self.y_prep_train = self.ss_y.fit_transform(
            self.y_train.reshape((-1, 1)))
        self.y_prep_test = self.ss_y.transform(self.y_test.reshape((-1,1)))

