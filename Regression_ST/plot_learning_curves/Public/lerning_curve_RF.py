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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import learning_curve

import plot_learning_curve
import load_data
import os

name_clf = 'LinearRegression'


#load data

data, labels = load_data.function_load_data()


outer_kf = KFold(n_splits=5, shuffle=True, random_state=2)


#clf 
pca = PCA(random_state=42)

regr_svml = LinearRegression()

clf=TransformedTargetRegressor(regressor=regr_svml,
                                     transformer=MinMaxScaler())

steps = [('scaler', StandardScaler()), ('red_dim', None), ('clf', clf)]

pipeline = Pipeline(steps)

title = "Learning_Curves_LinearRegression"

plot_learning_curve.function_plot_learning_curve(estimator=pipeline, features=data, target=labels, train_sizes= np.linspace(0.1, 1.0, 5),
                    cv=outer_kf, title=title)

#create folder and save

outname = f'{title}.png'

outdir = f'/home/leonardo/Scrivania/Presentazione/img_learning_curve/PA/ST/{name_clf}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

plt.savefig(fullname)
plt.close()

#clf optimized
pca = PCA(n_components=0.9, random_state=42)

regr_RF = RandomForestRegressor(n_estimators=100, max_depth=30, criterion='mae', random_state=503)

clf_opt=TransformedTargetRegressor(regressor=regr_RF,
                                     transformer=MinMaxScaler())

steps_opt = [('scaler', StandardScaler()), ('red_dim', None), ('clf', regr_RF)]

pipeline_opt = Pipeline(steps_opt)

title = "Learning_Curves_RandomForestClassifier_Optimized"

plot_learning_curve.function_plot_learning_curve(estimator=pipeline_opt, features=data, target=labels, train_sizes= np.linspace(0.1, 1.0, 5),
                    cv=outer_kf, title=title)

#train_sizes, train_scores, validation_scores = learning_curve(
#regr_RF, data, labels, train_sizes =
#np.linspace(0.1, 1.0, 5),
#cv = outer_kf, scoring = 'neg_mean_squared_error')
#train_scores_mean = -train_scores.mean(axis = 1)
#test_scores_mean = -validation_scores.mean(axis = 1)
#
#train_scores_std = np.std(train_scores, axis=1)
#test_scores_std = np.std(validation_scores, axis=1)



#create folder and save

outname = f'{title}.png'

outdir = f'/home/leonardo/Scrivania/Presentazione/img_learning_curve/PA/ST/{name_clf}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

plt.savefig(fullname)
plt.close()