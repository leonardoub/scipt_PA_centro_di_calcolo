#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:39:19 2020

@author: leonardo
"""


import pandas as pd

path = '/home/leonardo/Scrivania/result_PA/06_03/result_CV_bis/ST_regression/*/*/MAE/*/*.csv'

import glob
for name in glob.glob(path):
    #print(name)
    data = pd.read_csv(name) 

    MAE_train_mean = data['MAE_train'].mean()
    MAE_test_mean = data['MAE_test'].mean()
    
    MAE_train_std = data['MAE_train'].std()
    MAE_test_std = data['MAE_test'].std()


    df_train_MAE_mean = pd.DataFrame([{'MAE_train_mean':MAE_train_mean}])
    df_train_MAE_std = pd.DataFrame([{'MAE_train_std':MAE_train_std}])

    
    df_test_MAE_mean = pd.DataFrame([{'MAE_test_mean':MAE_test_mean}])
    df_test_MAE_std = pd.DataFrame([{'MAE_test_std':MAE_test_std}])


    df = pd.concat([data, df_train_MAE_mean, df_train_MAE_std, df_test_MAE_mean, df_test_MAE_std], axis=1)

    df.to_csv(name)


#data = pd.read_csv(name) 

#acc_train_mean = data['accuracy_train'].mean()

#data['accuracy_train'] = acc_train_mean


