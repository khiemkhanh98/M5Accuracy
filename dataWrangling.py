#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:34:01 2020

@author: duypham
@credit: https://www.kaggle.com/kneroma/m5-forecast-v2-python
"""

import numpy as np
import pandas as pd

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
h = 28 
max_lags = 366
tr_last = 1913

class dataWrangling():
    def __init__(self,folderName):
        self.folderName = folderName
        
    def toTable_(self,is_train = True, nrows = None, first_day = 1200):
        prices = pd.read_csv(self.folderName + "sell_prices.csv", dtype = PRICE_DTYPES)
        for col, col_dtype in PRICE_DTYPES.items():
            if col_dtype == "category":
                prices[col] = prices[col].cat.codes.astype("int16")
                prices[col] -= prices[col].min()
                
        cal = pd.read_csv(self.folderName + "calendar.csv", dtype = CAL_DTYPES)
        cal["date"] = pd.to_datetime(cal["date"])
        for col, col_dtype in CAL_DTYPES.items():
            if col_dtype == "category":
                cal[col] = cal[col].cat.codes.astype("int16")
                cal[col] -= cal[col].min()
        
        start_day = max(1 if is_train  else tr_last-max_lags, first_day)
        numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
        catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
        dtype = {numcol:"float32" for numcol in numcols} 
        dtype.update({col: "category" for col in catcols if col != "id"})
        dt = pd.read_csv(self.folderName + "sales_train_validation.csv", 
                         nrows = nrows, usecols = catcols + numcols, dtype = dtype)
        for col in catcols:
            if col != "id":
                dt[col] = dt[col].cat.codes.astype("int16")
                dt[col] -= dt[col].min()
        
        if not is_train:
            for day in range(tr_last+1, tr_last+ 2*h +1):
                dt[f"d_{day}"] = np.nan
        
        dt = pd.melt(dt,
                      id_vars = catcols,
                      value_vars = [col for col in dt.columns if col.startswith("d_")],
                      var_name = "d",
                      value_name = "sales")
        
        dt = dt.merge(cal, on= "d", copy = False)
        dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
        
        return dt
    
if __name__ == "__main__":
    FIRST_DAY = 800
    df = dataWrangling('./m5-forecasting-accuracy/').toTable_(is_train = True, first_day = FIRST_DAY)