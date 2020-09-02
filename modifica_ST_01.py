#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:11:54 2020

@author: leonardo
"""

import pandas as pd 
import os


train_dataset_path = '/home/leonardo/Scrivania/TESI_TOT/TESI_PA/data/database_training2.csv'
test_dataset_path = '/home/leonardo/Scrivania/TESI_TOT/TESI_PA/data/database_nostro.csv'

df_train = pd.read_csv(train_dataset_path)
df_test = pd.read_csv(test_dataset_path)



df_train.loc[df_train['Survival.time (months)'] < 24, 'Survival.time (months)'] = 0
df_train.loc[df_train['Survival.time (months)'] >= 24, 'Survival.time (months)'] = 1


df_test.loc[df_test['Survival.time (months)'] < 24, 'Survival.time (months)'] = 0
df_test.loc[df_test['Survival.time (months)'] >= 24, 'Survival.time (months)'] = 1



outname = f'database_training2_ST01.csv'
outdir = f'/home/leonardo/Scrivania/TESI_TOT/TESI_PA/data'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

df_train.to_csv(fullname)



outname = f'database_nostro_ST01.csv'
outdir = f'/home/leonardo/Scrivania/TESI_TOT/TESI_PA/data'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

df_test.to_csv(fullname)

