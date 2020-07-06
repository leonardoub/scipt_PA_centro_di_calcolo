#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 22:21:16 2020

@author: leonardo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import load_data_ST
import os

name_clf = 'RandomForestRegressor'


#load data

pu_data, pu_labels, pa_data, pa_labels  = load_data_ST.function_load_data_ST()


regr=RandomForestRegressor(n_estimators=100, max_depth=10, criterion='mae', random_state=503)

 
clf = TransformedTargetRegressor(regressor=regr,
                                     transformer=MinMaxScaler())


#RandomForestRegressor
steps = [('scaler', MinMaxScaler()), ('red_dim', PCA(n_components=0.85)), ('clf', clf)]

pipeline = Pipeline(steps)

pipeline.fit(pu_data, pu_labels)

train_mae = pipeline.score(pu_data, pu_labels)
test_mae = pipeline.score(pa_data, pa_labels)

pred_train = pipeline.predict(pu_data)
pred_test = pipeline.predict(pa_data)

dict_train = {'True Survival Time in months': pu_labels, 'Predicted Survival Time in months': pred_train} 
df_train = pd.DataFrame(dict_train)

dict_test = {'True Survival Time in months': pa_labels, 'Predicted Survival Time in months': pred_test} 
df_test = pd.DataFrame(dict_test)


df_train_sorted = df_train.sort_values('Predicted Survival Time in months',ascending=True)
df_test_sorted = df_test.sort_values('Predicted Survival Time in months',ascending=True)


fig4, ax4 = plt.subplots()

df_test_sorted.plot(ax=ax4, x='True Survival Time in months', y='Predicted Survival Time in months',
                    kind='scatter', fontsize=10, 
                    xticks=np.arange(0, 55, 5), yticks=np.arange(0, 55, 5))

ax4.set_xlim((0,55))
ax4.set_ylim((0,55))


ax4.set_aspect('equal', adjustable='box')


ax4.set_title(f'Test')


fig4.set_figwidth(6)
fig4.set_figheight(6)

##create folder and save
#
#outname = f'ST_train_PU_test_PA_RF_PSTvsTST.pdf'
#
#outdir = f'/home/leonardo/Scrivania/scrittura_TESI/img/original_create_da_me_python/PA/ST/train_PU_test_PA'
#if not os.path.exists(outdir):
#    os.makedirs(outdir)
#
#fullname = os.path.join(outdir, outname)    
#
#plt.savefig(fullname)
#plt.close()

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

fig5, ax5 = plt.subplots()

df_train_sorted.plot(ax=ax5, x='True Survival Time in months', y='Predicted Survival Time in months',
                    kind='scatter', fontsize=10, xticks=np.arange(0, 80, 5), yticks=np.arange(0, 80, 5))

ax5.set_xlim((0,80))
ax5.set_ylim((0,80))


ax5.set_aspect('equal', adjustable='box')

ax5.set_title(f'Train')


fig5.set_figwidth(6)
fig5.set_figheight(6)

##create folder and save
#
#outname = f'ST_train_PU_RF_PSTvsTST.pdf'
#
#outdir = f'/home/leonardo/Scrivania/scrittura_TESI/img/original_create_da_me_python/PA/ST/train_PU_test_PA'
#if not os.path.exists(outdir):
#    os.makedirs(outdir)
#
#fullname = os.path.join(outdir, outname)    
#
#plt.savefig(fullname)
#plt.close()




#n_train=np.arange(1,132)
#
#fig, ax1 = plt.subplots()
#    
#ax1.scatter(n_train, pu_labels, color = "green", s=2)    
#ax1.scatter(n_train, pred_train, color = "red", s=2)    
#
#ax1.vlines(x, 0, y, linestyle="dashed")
#ax1.hlines(y, 0, x, linestyle="dashed")
#plt.scatter(x, y, zorder=2)

#n_test=np.arange(1,48)
#
#fig2, ax2 = plt.subplots()
#
#legend_labels=['True ST','Predicted ST']
#x_ticks_labels=[f'{i}' for i in n_test]
#
#ax2.scatter(n_test, pa_labels, marker='X', color = "green", s=5, label = 'True ST')    
#ax2.scatter(n_test, pred_test, marker='X', color = "red", s=5, label = 'Predicted ST')  
#
#ax2.legend()
#
#ax2.vlines(n_test, 0, pa_labels, color='gray', linestyle="dotted")
#ax2.vlines(n_test, 0, pred_test, color='gray', linestyle="dotted")
#
#ax2.set_xticks(n_test)
#ax2.set_xticklabels(x_ticks_labels, fontsize=7)
#
#ax2.set_yticks(np.arange(0,55,5))
#
#ax2.set_xlabel('Patient')
#ax2.set_ylabel('Survival time in months')
#
#
#fig2.set_figwidth(8)
#fig2.set_figheight(6)







##create folder and save
#
#outname = f'ST_train_PU_test_PA_RF.png'
#
#outdir = f'/home/leonardo/Scrivania/scrittura_TESI/img/original_create_da_me_python/PA/ST/'
#if not os.path.exists(outdir):
#    os.makedirs(outdir)
#
#fullname = os.path.join(outdir, outname)    
#
#plt.savefig(fullname)
#plt.close()




#fig3, ax3 = plt.subplots()
#
#df_test_sorted.plot(ax=ax3, x='Predicted Survival Time in months', y='True Survival Time in months', kind='scatter', fontsize=6)
#
#ax3.set_xticks(np.arange(5,55,5))
#ax3.set_yticks(np.arange(5,55,5))
#
#fig3.set_figwidth(8)
#fig3.set_figheight(6)

