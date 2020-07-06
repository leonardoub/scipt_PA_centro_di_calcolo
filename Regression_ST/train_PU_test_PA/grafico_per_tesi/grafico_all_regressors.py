#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 06:37:30 2020

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
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR 

import load_data_ST
import os

regr_RF_name = 'RandomForestRegressor'
regr_svml_name = 'SVR linear'
regr_rbf_name = 'SVR rbf'
regr_sig_name = 'SVR sigmoid'

#load data

pu_data, pu_labels, pa_data, pa_labels  = load_data_ST.function_load_data_ST()


regr_RF = RandomForestRegressor(n_estimators=100, max_depth=10, criterion='mae', random_state=503)

regr_svml = LinearRegression()

regr_rbf = SVR(kernel='rbf', C=0.25, gamma=0.0078125)

regr_sig = SVR(kernel='sigmoid', C=0.0625, gamma=125)

 
clf_RF = TransformedTargetRegressor(regressor=regr_RF,
                                     transformer=MinMaxScaler())

clf_svml = TransformedTargetRegressor(regressor=regr_svml,
                                     transformer=MinMaxScaler())

clf_rbf = TransformedTargetRegressor(regressor=regr_rbf,
                                     transformer=MinMaxScaler())

clf_sig = TransformedTargetRegressor(regressor=regr_sig,
                                     transformer=MinMaxScaler())

#RandomForestRegressor
steps = [('scaler', MinMaxScaler()), ('red_dim', PCA(n_components=0.85)), ('clf', clf_RF)]

pipeline = Pipeline(steps)

pipeline.fit(pu_data, pu_labels)

train_mae = pipeline.score(pu_data, pu_labels)
test_mae = pipeline.score(pa_data, pa_labels)

pred_train = pipeline.predict(pu_data)
pred_test = pipeline.predict(pa_data)

dict_train_RF = {'True Survival Time in months': pu_labels, 'Predicted Survival Time in months': pred_train} 
df_train_RF = pd.DataFrame(dict_train_RF)

dict_test_RF = {'True Survival Time in months': pa_labels, 'Predicted Survival Time in months': pred_test} 
df_test_RF = pd.DataFrame(dict_test_RF)


df_train_sorted_RF = df_train_RF.sort_values('Predicted Survival Time in months',ascending=True)
df_test_sorted_RF = df_test_RF.sort_values('Predicted Survival Time in months',ascending=True)


#SVRlinear
steps = [('scaler', MinMaxScaler()), ('red_dim', PCA(n_components=0.85)), ('clf', clf_svml)]

pipeline = Pipeline(steps)

pipeline.fit(pu_data, pu_labels)

train_mae = pipeline.score(pu_data, pu_labels)
test_mae = pipeline.score(pa_data, pa_labels)

pred_train = pipeline.predict(pu_data)
pred_test = pipeline.predict(pa_data)

dict_train_svml = {'True Survival Time in months': pu_labels, 'Predicted Survival Time in months': pred_train} 
df_train_svml = pd.DataFrame(dict_train_svml)

dict_test_svml = {'True Survival Time in months': pa_labels, 'Predicted Survival Time in months': pred_test} 
df_test_svml = pd.DataFrame(dict_test_svml)


df_train_sorted_svml = df_train_svml.sort_values('Predicted Survival Time in months',ascending=True)
df_test_sorted_svml = df_test_svml.sort_values('Predicted Survival Time in months',ascending=True)



#SVRrbf
steps = [('scaler', MinMaxScaler()), ('red_dim', PCA(n_components=0.85)), ('clf', clf_rbf)]

pipeline = Pipeline(steps)

pipeline.fit(pu_data, pu_labels)

train_mae = pipeline.score(pu_data, pu_labels)
test_mae = pipeline.score(pa_data, pa_labels)

pred_train = pipeline.predict(pu_data)
pred_test = pipeline.predict(pa_data)

dict_train_rbf = {'True Survival Time in months': pu_labels, 'Predicted Survival Time in months': pred_train} 
df_train_rbf = pd.DataFrame(dict_train_rbf)

dict_test_rbf = {'True Survival Time in months': pa_labels, 'Predicted Survival Time in months': pred_test} 
df_test_rbf = pd.DataFrame(dict_test_rbf)


df_train_sorted_rbf = df_train_rbf.sort_values('Predicted Survival Time in months',ascending=True)
df_test_sorted_rbf = df_test_rbf.sort_values('Predicted Survival Time in months',ascending=True)




#SVRsig
steps = [('scaler', MinMaxScaler()), ('red_dim', PCA(n_components=0.85)), ('clf', clf_sig)]

pipeline = Pipeline(steps)

pipeline.fit(pu_data, pu_labels)

train_mae = pipeline.score(pu_data, pu_labels)
test_mae = pipeline.score(pa_data, pa_labels)

pred_train = pipeline.predict(pu_data)
pred_test = pipeline.predict(pa_data)

dict_train_sig = {'True Survival Time in months': pu_labels, 'Predicted Survival Time in months': pred_train} 
df_train_sig = pd.DataFrame(dict_train_sig)

dict_test_sig = {'True Survival Time in months': pa_labels, 'Predicted Survival Time in months': pred_test} 
df_test_sig = pd.DataFrame(dict_test_sig)


df_train_sorted_sig = df_train_sig.sort_values('Predicted Survival Time in months',ascending=True)
df_test_sorted_sig = df_test_sig.sort_values('Predicted Survival Time in months',ascending=True)




#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################


min_test = 0
max_test = 55
step_test = 5




fig_test, axs = plt.subplots(2, 2)

df_test_sorted_RF.plot(ax=axs[0,0], x='True Survival Time in months', y='Predicted Survival Time in months',
                    kind='scatter', fontsize=10, 
                    xticks=np.arange(min_test, max_test, step_test), yticks=np.arange(min_test, max_test, step_test))

axs[0,0].set_xlim((min_test, max_test))
axs[0,0].set_ylim((min_test, max_test))
axs[0,0].set_aspect('equal', adjustable='box')
axs[0,0].set_title(regr_RF_name)


df_test_sorted_svml.plot(ax=axs[0,1], x='True Survival Time in months', y='Predicted Survival Time in months',
                    kind='scatter', fontsize=10, 
                    xticks=np.arange(min_test, max_test, step_test), yticks=np.arange(min_test, max_test, step_test))

axs[0,1].set_xlim((min_test, max_test))
axs[0,1].set_ylim((min_test, max_test))
axs[0,1].set_aspect('equal', adjustable='box')
axs[0,1].set_title(regr_svml_name)


df_test_sorted_rbf.plot(ax=axs[1,0], x='True Survival Time in months', y='Predicted Survival Time in months',
                    kind='scatter', fontsize=10, 
                    xticks=np.arange(min_test, max_test, step_test), yticks=np.arange(min_test, max_test, step_test))

axs[1,0].set_xlim((min_test, max_test))
axs[1,0].set_ylim((min_test, max_test))
axs[1,0].set_aspect('equal', adjustable='box')
axs[1,0].set_title(regr_rbf_name)


df_test_sorted_sig.plot(ax=axs[1,1], x='True Survival Time in months', y='Predicted Survival Time in months',
                    kind='scatter', fontsize=10, 
                    xticks=np.arange(min_test, max_test, step_test), yticks=np.arange(min_test, max_test, step_test))

axs[1,1].set_xlim((min_test, max_test))
axs[1,1].set_ylim((min_test, max_test))
axs[1,1].set_aspect('equal', adjustable='box')
axs[1,1].set_title(regr_sig_name)


fig_test.set_figwidth(10)
fig_test.set_figheight(10)

fig_test.suptitle('TEST', fontsize=14, y=1)

fig_test.tight_layout()

#create folder and save

outname = f'ST_train_PU_test_PA_ALL_REGRESSORS_PSTvsTST.pdf'

outdir = f'/home/leonardo/Scrivania/scrittura_TESI/img/original_create_da_me_python/PA/ST/train_PU_test_PA'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

plt.savefig(fullname)
plt.close()

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################


min_train = 0
max_train = 85
step_train = 10



fig_train, axs = plt.subplots(2, 2)

df_train_sorted_RF.plot(ax=axs[0,0], x='True Survival Time in months', y='Predicted Survival Time in months',
                    kind='scatter', fontsize=10, 
                    xticks=np.arange(min_train, max_train, step_train), yticks=np.arange(min_train, max_train, step_train))

axs[0,0].set_xlim((min_train, max_train,))
axs[0,0].set_ylim((min_train, max_train,))
axs[0,0].set_aspect('equal', adjustable='box')
axs[0,0].set_title(regr_RF_name)


df_train_sorted_svml.plot(ax=axs[0,1], x='True Survival Time in months', y='Predicted Survival Time in months',
                    kind='scatter', fontsize=10, 
                    xticks=np.arange(min_train, max_train, step_train), yticks=np.arange(min_train, max_train, step_train))

axs[0,1].set_xlim((min_train, max_train,))
axs[0,1].set_ylim((min_train, max_train,))
axs[0,1].set_aspect('equal', adjustable='box')
axs[0,1].set_title(regr_svml_name)


df_train_sorted_rbf.plot(ax=axs[1,0], x='True Survival Time in months', y='Predicted Survival Time in months',
                    kind='scatter', fontsize=10, 
                    xticks=np.arange(min_train, max_train, step_train), yticks=np.arange(min_train, max_train, step_train))

axs[1,0].set_xlim((min_train, max_train,))
axs[1,0].set_ylim((min_train, max_train,))
axs[1,0].set_aspect('equal', adjustable='box')
axs[1,0].set_title(regr_rbf_name)


df_train_sorted_sig.plot(ax=axs[1,1], x='True Survival Time in months', y='Predicted Survival Time in months',
                    kind='scatter', fontsize=10, 
                    xticks=np.arange(min_train, max_train, step_train), yticks=np.arange(min_train, max_train, step_train))

axs[1,1].set_xlim((min_train, max_train,))
axs[1,1].set_ylim((min_train, max_train,))
axs[1,1].set_aspect('equal', adjustable='box')
axs[1,1].set_title(regr_sig_name)


fig_train.set_figwidth(10)
fig_train.set_figheight(10)

fig_train.suptitle('TRAIN', fontsize=14, y=1)

fig_train.tight_layout()

#create folder and save

outname = f'ST_train_PU_ALL_REGRESSORS_PSTvsTST.pdf'

outdir = f'/home/leonardo/Scrivania/scrittura_TESI/img/original_create_da_me_python/PA/ST/train_PU_test_PA'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

plt.savefig(fullname)
plt.close()


