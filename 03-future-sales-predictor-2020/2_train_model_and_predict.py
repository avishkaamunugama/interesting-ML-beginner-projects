import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import metrics
import joblib


# load preprocessed data
data = joblib.load("data.dat")
item_sales_mean = joblib.load("item_sales_mean.dat")

# split to x_train y_train x_test y_test 
x = data.iloc[:,[0,1,3,4,6]]
y = data.iloc[:,5]

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2, random_state = 18)

# normalise and clip the results to meet copition rules
y_train = np.log1p(y_train.clip(0.,20.))
y_test = np.log1p(y_test.clip(0.,20.))


# train model using ExtraTreesRegressor
regressor = ExtraTreesRegressor(n_estimators=60,n_jobs=-1,max_depth = 12,random_state=18)
regressor.fit(x_train,y_train)

# print rmse score of the trained model
y_pred = (regressor.predict(x_test)).clip(0.,20.)
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# create submission using the given data
test_data = pd.read_csv('data/test.csv')

# since we are asked to predict sales for the month following the final month in dataset
test_data['month'] = 11
test_data['year'] = 2015
test_data['item_cnt_month'] = 0

# adding the previously calculated mean
test_data = pd.merge(test_data , item_sales_mean , on=['shop_id','item_id','month'] , how='left').fillna(0.0)
test_data_x = test_data.iloc[:,[1,2,3,4,6]]

# create submission.csv
test_data['item_cnt_month'] = regressor.predict(test_data_x).clip(0.,20.)
test_data['item_cnt_month'] = np.expm1(test_data['item_cnt_month'])
test_data[['ID','item_cnt_month']].to_csv('submission.csv',index = False)

