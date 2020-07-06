#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:14:16 2020

@author: leonardo
"""

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

path_pupa = '/home/leonardo/Scrivania/result_PA/06_03/result_CV_bis/2_classes_OS/Train_PU_Test_PA_OS_1_2/*/*.csv'
path_papu = '/home/leonardo/Scrivania/result_PA/06_03/result_CV_bis/2_classes_OS/Train_PA_Test_PU_OS_1_2/*/*.csv'

import glob

clf_pupa_list = [] 
clf_papu_list = [] 

#D = {'clf':[], 'stat':[], 'pvalue':[]}
D={}

for name_pupa, name_papu in zip(sorted(glob.glob(path_pupa)), sorted(glob.glob(path_papu))):
    #print(name)
    clf_pupa = os.path.split(name_pupa)[-1]
    clf_pupa = clf_pupa[12:-28]
    clf_pupa_list.append(clf_pupa)
    
    clf_papu = os.path.split(name_papu)[-1]
    clf_papu = clf_papu[12:-28]
    clf_papu_list.append(clf_papu)
    
    
    data_pupa = pd.read_csv(name_pupa) 
    data_papu = pd.read_csv(name_papu)
    
    roc_auc_test_mean_pupa = data_pupa['test_loop_roc_auc_scores_predict_proba']
    roc_auc_test_mean_papu = data_papu['test_loop_roc_auc_scores_predict_proba']


    stat, pvalue = stats.ttest_ind(roc_auc_test_mean_pupa, roc_auc_test_mean_papu, equal_var = False) #t test con equal_var=False è il Welch's t-test
    
    D[clf_pupa]=pvalue

