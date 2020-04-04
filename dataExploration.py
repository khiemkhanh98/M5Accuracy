#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:20:20 2020

@author: duypham
"""

import pandas as pd
import os

class dataExploration:
    def exploring_data(fileName):
        data = pd.read_csv(fileName)
        print(data.head())
        columns_num = len(data.keys())
        rows_num = len(data.values)
        print('The Calendar data has {} data columns and {} records'.format(columns_num, rows_num))
        [print(i) for i in data.keys()]
        na_count = 0
        na_pos = []
        for i in range(len(data.values)):
            row = data.values[i]
            for j in range(len(row)):
                if pd.isna(row[j]):
                    na_count += 1
                    na_pos.append(str(i)+' '+str(j))
        return print('The number of NaN values is {}. Check for details in na_pos variable in the code.'
                     .format(na_count))
                    

if __name__ == "__main__":
    dataFiles = [i for i in os.listdir(os.path.join('m5-forecasting-accuracy')) if i != '.DS_Store']
    for fileName in dataFiles:
        dataExploration.exploring_data('./m5-forecasting-accuracy/'+ fileName)