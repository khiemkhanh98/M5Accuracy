# adapted from https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50/ and https://www.kaggle.com/ragnar123/very-fst-model

from memoryReduction import *
from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn import preprocessing, metrics


def create_dt(is_train = True, nrows = None, first_day = 800):
    CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
    PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

    h = 28 
    max_lags = 30
    tr_last = 1913
    fday = datetime(2016,4, 25) 
    
    prices = pd.read_csv("m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv("./m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    print(len(numcols))
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv("./m5-forecasting-accuracy/sales_train_validation.csv", nrows = nrows, usecols = catcols + numcols, dtype = dtype)
 
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    del prices,cal
    gc.collect()
    dt = reduce_mem_usage(dt)
    return dt

def create_fea(dt):
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