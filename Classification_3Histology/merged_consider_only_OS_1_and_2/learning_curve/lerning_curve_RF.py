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
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score

import plot_learning_curve
import load_data_3_class
import os

name_clf = 'RandomForestClassifier'


#load data

data, labels = load_data_3_class.function_load_data_3_class()


#Vettorizzare i label
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)


#clf 
pca = PCA(random_state=42)

clf=RandomForestClassifier(random_state=503)

steps = [('scaler', StandardScaler()), ('red_dim', None), ('clf', clf)]

pipeline = Pipeline(steps)

title = "Learning_Curves_RandomForestClassifier"

plot_learning_curve.function_plot_learning_curve(pipeline, title, data, labels_encoded, ylim=(0.0, 1.01),
                    cv=outer_kf, n_jobs=1)

#create folder and save

outname = f'{title}.png'

outdir = f'/home/leonardo/Scrivania/Presentazione/img_learning_curve/PA/3H/MERGED_only_OS12/{name_clf}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

plt.savefig(fullname)
plt.close()

#clf optimized
pca = PCA(n_components=0.9, random_state=42)

clf_opt=RandomForestClassifier(criterion='entropy', n_estimators=50, max_depth=10, random_state=503)

steps_opt = [('scaler', StandardScaler()), ('red_dim', None), ('clf', clf_opt)]

pipeline_opt = Pipeline(steps_opt)

title = "Learning_Curves_RandomForestClassifier_Optimized"

plot_learning_curve.function_plot_learning_curve(pipeline_opt, title, data, labels_encoded, ylim=(0.0, 1.01),
                    cv=outer_kf, n_jobs=1)


#create folder and save

outname = f'{title}.png'

outdir = f'/home/leonardo/Scrivania/Presentazione/img_learning_curve/PA/3H/MERGED_only_OS12/{name_clf}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

plt.savefig(fullname)
plt.close()