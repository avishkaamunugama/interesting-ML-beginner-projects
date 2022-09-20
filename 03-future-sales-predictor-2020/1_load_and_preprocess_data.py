import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import metrics
import joblib

# importing data 
data = pd.read_csv('data/sales_train.csv')

# preprocessing data
# extract month and year from date
data['date'] = pd.to_datetime(data['date'] , format = "%d.%m.%Y")
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# calculate total sales made for each item per month 
data = data.groupby(['year','month','date_block_num','shop_id','item_id'], as_index=False)['item_cnt_day'].sum().rename(columns={'item_cnt_day':'item_cnt_month'})

# calculate mean number sales for each item per month
item_sales_mean =  data[['shop_id','item_id','month','item_cnt_month']].groupby(['month','shop_id','item_id'] , as_index=False)['item_cnt_month'].mean().rename(columns={'item_cnt_month':'monthly_item_mean'})
data = pd.merge(data , item_sales_mean , on=['shop_id','item_id','month'] , how='left').fillna(0.0)


# Save processed data
joblib.dump(data, "data.dat")
joblib.dump(item_sales_mean, "item_sales_mean.dat")

