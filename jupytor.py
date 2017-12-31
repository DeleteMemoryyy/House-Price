import dataloader
from dataloader import DataConfig, DataLoader
import pandas as pd
import numpy as np

dl = DataLoader(DataConfig())

train_file = 'data/trainData.csv'
test_file = 'data/testData.csv'
all_file = 'data/allData.csv'
other_columns = ['Id', 'SalePrice', 'TestLabel']

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
all_data = pd.read_csv(all_file)
proc_data = pd.concat(
    [train_data, test_data], ignore_index=True)
proc_data['SalePrice'] = proc_data['SalePrice'].fillna(0)
proc_data['TestLabel'] = proc_data['SalePrice'] == 0

# ser = all_data.loc[0]
# ser.drop(['SalePrice','Id'])
# all(ser == ser)
# all(ser.dropna().values == ser.dropna().values)

# test_result_list = []
# def fill_in_test(x):
#     print(len(test_result_list))
#     temp_this_row = x.drop(['Id'])
#     for idx in all_data.index:
#         temp_row = all_data.loc[idx].drop(['SalePrice', 'Id'])
#         val_this = temp_this_row.dropna().values
#         val_temp = temp_row.dropna().values
#         flag = len(val_this) == len(val_temp) and all(val_this == val_temp)

#         if flag:
#             print('Yes',idx)
#             test_result_list.append(all_data.loc[idx]['SalePrice'])

# new_proc_data = test_data.apply(fill_in_test,axis=1)
# len(test_result_list)

# df = pd.DataFrame(test_result_list,columns=['SalePrice'])
# df.to_csv('data/testResult.csv',index=False)

dl.y_train
dl.y_log_trainxx'ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssccccccc