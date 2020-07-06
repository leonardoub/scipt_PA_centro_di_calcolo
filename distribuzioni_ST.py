#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 23:53:18 2020

@author: leonardo
"""


import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np


train_dataset_path = '/home/leonardo/Scrivania/TESI_PA/data/database_training2.csv'
test_dataset_path = '/home/leonardo/Scrivania/TESI_PA/data/database_nostro.csv'

df_train = pd.read_csv(train_dataset_path)
df_test = pd.read_csv(test_dataset_path)

df_train.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)
df_test.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)

df_train.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)
df_test.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)

#public_data = df_train.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)
#PA_data = df_test.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)
#
public_labels = df_train.Surv_time_months
PA_labels = df_test.Surv_time_months

reference = np.arange(0,90,10)
reference_y = np.arange(0,22,2)


fig, axs = plt.subplots(1, 2, sharey=True)

c='royalblue'

axs[0].hist(public_labels, bins=20, range=(0, 80))
#axs[0].set_xticklabels(reference)
axs[0].set_xlabel('Survival time [months]')
axs[0].set_yticks(reference_y)
axs[0].set_ylabel('Counts')



axs[1].hist(PA_labels, bins=20, range=(0, 80))
axs[1].set_xlabel('Survival time [months]')
#axs[1].set_yticks(reference_y)
axs[1].set_yticklabels(reference_y)
axs[1].set_ylabel('Counts')
axs[1].yaxis.set_tick_params(labelbottom=True)


fig.set_figwidth(15)
fig.set_figheight(4)





#create folder and save

outname = f'ST_distribution.pdf'

outdir = f'/home/leonardo/Scrivania/scrittura_TESI/img/original_create_da_me_python/PA/ST/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

plt.savefig(fullname)
plt.close()


