
import pandas as pd
import dataWrangling
from memoryReduction import *
import gc
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn import preprocessing, metrics
import multiprocessing as mp
import threading


    
def create_fea1(dt):
    print("begin")
    lags = [7, 28, 29, 30,31,90,365]
    #lags = [7]
    lag_cols = [f"lag_{lag}" for lag in lags]
    price_lag = [f"price_lag_{lag}" for lag in lags]
    for lag, lag_col, price_lag_col in zip(lags, lag_cols, price_lag):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)
        dt[price_lag_col] = dt[["id","sell_price"]].groupby("id")["sell_price"].shift(lag)
    
    wins = lags
    for i, lag_data in enumerate(zip(lags, lag_cols)):
        lag = lag_data[0]
        lag_col = lag_data[1]
        print(lag,lag_col)
        dt[f"rmean_{lag}_{wins[i]}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(wins[i]).mean())
        dt[f"rstd_{lag}_{wins[i]}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(wins[i]).std())
        dt[f"rskew_{lag}_{wins[i]}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(wins[i]).skew())
        dt[f"rskurt_{lag}_{wins[i]}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(wins[i]).kurt())
    
    dt['date'] = pd.to_datetime(dt['date'])
    dt['year'] = dt['date'].dt.year
    dt['month'] = dt['date'].dt.month
    dt['week'] = dt['date'].dt.week
    dt['day'] = dt['date'].dt.day
    dt['dayofweek'] = dt['date'].dt.dayofweek
    
    cat = ['year','month','week','day','dayofweek']
    encoder = preprocessing.LabelEncoder()
    for feature in cat:
        dt[feature] = encoder.fit_transform(dt[feature])
        
    #dt.dropna(inplace = True)
    return dt

if __name__ == '__main__':
    pool = mp.Pool(4)
    df = dataWrangling.create_dt(True, 1)
    df = pool.map(create_fea1, df)
    df = reduce_mem_usage(df)
    df.to_pickle("data.pkl")
