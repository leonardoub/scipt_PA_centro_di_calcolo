#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:27:48 2020

@author: leonardo
"""

from scipy import stats
import os
import numpy as np
import pandas as pd

path_pu = '/home/leonardo/Scrivania/result_PA/06_03/result_CV_bis/3_classes_H/PUBLIC/*/*/*/*.csv'
path_merged = '/home/leonardo/Scrivania/result_PA/06_03/result_CV_bis/3_classes_H/merged_consider_only_OS_1_and_2/*/*/*/*.csv'

import glob

clf_pu_list = [] 
clf_merged_list = [] 

#D = {'clf':[], 'stat':[], 'pvalue':[]}
D={}

for name_pu, name_merged in zip(sorted(glob.glob(path_pu)), sorted(glob.glob(path_merged))):
    #print(name)
    clf_pu = os.path.split(name_pu)[-1]
    clf_pu = clf_pu[12:-17]
    clf_pu_list.append(clf_pu)
    
    clf_merged = os.path.split(name_merged)[-1]
    clf_merged = clf_merged[12:-14]
    clf_merged_list.append(clf_merged)
    
    
    data_pu = pd.read_csv(name_pu) 
    data_merged = pd.read_csv(name_merged)
    
    roc_auc_test_mean_pu = data_pu['outer_loop_roc_auc_ovr_weigthed_scores_predict_proba']
    roc_auc_test_mean_merged = data_merged['outer_loop_roc_auc_ovr_weigthed_scores_predict_proba']


    stat, pvalue = stats.ttest_ind(roc_auc_test_mean_pu, roc_auc_test_mean_merged, equal_var = False) #t test con equal_var=False è il Welch's t-test
    
    D[clf_pu]=pvalue

