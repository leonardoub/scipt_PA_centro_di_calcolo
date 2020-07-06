#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:56:23 2020

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
#
#public_labels = public_labels.astype('float')
#PA_labels = PA_labels.astype('float')
#
#tot_data = pd.concat([public_data, PA_data], axis=0)
#tot_label = pd.concat([public_labels, PA_labels], axis=0)



my_dict = {'Lung1': public_labels, 'L-RT':PA_labels}

fig, ax = plt.subplots()

c='royalblue'

ax.boxplot(my_dict.values(), showmeans=True, whis='range',
            boxprops=dict( color=c),
            capprops=dict(color='black'),
            whiskerprops=dict(color=c),
            medianprops=dict(color='g'))
ax.set_xticklabels(my_dict.keys())


reference = np.arange(0,80,10)
left, right = plt.xlim()
ax.hlines(reference, xmin=left, xmax=right, color='gray', linestyles='--', linewidth=0.5)

ax.set_ylabel('Survival time [months]')


fig.set_figwidth(7)
fig.set_figheight(5)

#create folder and save

outname = f'ST_boxplot.pdf'

outdir = f'/home/leonardo/Scrivania/scrittura_TESI/img/original_create_da_me_python/PA/ST/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

plt.savefig(fullname)
plt.close()



#
#
#
#a1=df_train.boxplot(ax=ax, column=['Surv_time_months'], showmeans=True, whis='range', return_type='axes')
#df_test.boxplot(ax=a1, column=['Surv_time_months'], showmeans=True, whis='range')
#
#
#
#boxplot_pu.set_yticks(np.arange(0, 81,10))
#boxplot_pu.set_ylabel('Survival time [months]')
#plt.title('Public dataset')
#
#plt.figure(figsize=(5, 7))

mean_ST_PU = np.mean(public_labels)
std_ST_PU = np.std(public_labels)
median_ST_PU = np.median(public_labels)
min_ST_PU = np.min(public_labels)
max_ST_PU = np.max(public_labels)


mean_ST_PA = np.mean(PA_labels)
std_ST_PA = np.std(PA_labels)
median_ST_PA = np.median(PA_labels)
min_ST_PA = np.min(PA_labels)
max_ST_PA = np.max(PA_labels)

