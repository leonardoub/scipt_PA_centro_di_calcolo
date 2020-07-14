#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 08:45:56 2020

@author: leonardo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold, learning_curve
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score
import seaborn as sns

import load_data_ST
import os

name_clf = 'RandomForestRegressor'


#load data

pu_data, pu_labels, PA_data, PA_labels = load_data_ST.function_load_data_ST()


# Designate distributions to sample hyperparameters from 
n_features_to_test = [0.85, 0.9, 0.95]
n_tree = [10, 13, 18, 25, 33, 45, 60, 81, 100]
depth = [5, 10, 15, 20, 30, 45, 60, 75]


regr_RF = RandomForestRegressor(criterion='mae', random_state=503)

pca = PCA(random_state=42, n_components=0.85)

#clf 

#clf=TransformedTargetRegressor(regressor=regr_RF, transformer=MinMaxScaler())

steps = [('scaler', MinMaxScaler()), ('red_dim', pca), ('clf', regr_RF)]

pipeline = Pipeline(steps)



parameteres = [{'clf__regressor__n_estimators':n_tree, 'clf__regressor__max_depth':depth}]

outer_kf = KFold(n_splits=5, shuffle=True, random_state=2)

rf_gridsearch = GridSearchCV(estimator=clf, param_grid=parameteres, n_jobs=4, 
                             cv=outer_kf, return_train_score=True)

rf_gridsearch.fit(pu_data, pu_labels)

# and after some hours...
df_gridsearch = pd.DataFrame(rf_gridsearch.cv_results_)


max_scores = df_gridsearch.groupby(['param_n_estimators', 
                                    'param__max_depth']).max()
max_scores = max_scores.unstack()[['mean_test_score', 'mean_train_score']]
sns_plot = sns.heatmap(max_scores.mean_test_score, annot=True, fmt='.4g')





#create folder and save


outname = f'best_params_{name_clf}_PU_PA_regression_ST.csv'
outdir = f'/home/users/ubaldi/TESI_PA/fig_heatmap/ST_regression/train_PU_test_PA/{name_clf}'

if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

sns_plot.savefig("output.png")

