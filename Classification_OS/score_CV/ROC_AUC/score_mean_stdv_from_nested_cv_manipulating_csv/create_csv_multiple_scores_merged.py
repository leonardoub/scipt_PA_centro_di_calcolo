#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:14:38 2020

@author: leonardo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:48:35 2020

@author: leonardo
"""


import os
import pandas as pd



    
path = f'/home/leonardo/Scrivania/result_PA/06_03/result_CV_bis/2_classes_OS/Merged_OS_1_2/*/*.csv'
    
    
    
my_dict = {'ACC_TRAIN_MEAN': [],
               'ACC_TRAIN_STD': [],
               'ACC_TEST_MEAN': [],
               'ACC_TEST_STD': [], 
               'BAL_ACC_TRAIN_MEAN': [],
               'BAL_ACC_TRAIN_STD': [],
               'BAL_ACC_TEST_MEAN': [],
               'BAL_ACC_TEST_STD': [],
               'ROC_AUC_TRAIN_MEAN': [],
               'ROC_AUC_TRAIN_STD': [],
               'ROC_AUC_TEST_MEAN': [],
               'ROC_AUC_TEST_STD': []}
    
    
clf_list = []

import glob
for name in sorted(glob.glob(path)):
    print(name)
    data = pd.read_csv(name) 
    clf = os.path.split(name)[-1]
    clf = clf[12:-4]
#    my_dict['SCALER'].append(data['SCALER'][0])
    my_dict['ACC_TRAIN_MEAN'].append(data['train_accuracy_MEAN'][0])
    my_dict['ACC_TRAIN_STD'].append(data['train_accuracy_STD'][0])
    my_dict['ACC_TEST_MEAN'].append(data['test_accuracy_MEAN'][0])
    my_dict['ACC_TEST_STD'].append(data['test_accuracy_STD'][0])
    my_dict['BAL_ACC_TRAIN_MEAN'].append(data['train_balanced_accuracy_MEAN'][0])
    my_dict['BAL_ACC_TRAIN_STD'].append(data['train_balanced_accuracy_STD'][0])
    my_dict['BAL_ACC_TEST_MEAN'].append(data['test_balanced_accuracy_MEAN'][0])
    my_dict['BAL_ACC_TEST_STD'].append(data['test_balanced_accuracy_STD'][0])
    my_dict['ROC_AUC_TRAIN_MEAN'].append(data['train_ROC_AUC_score_MEAN'][0])
    my_dict['ROC_AUC_TRAIN_STD'].append(data['train_ROC_AUC_score_STD'][0])
    my_dict['ROC_AUC_TEST_MEAN'].append(data['test_ROC_AUC_score_MEAN'][0])
    my_dict['ROC_AUC_TEST_STD'].append(data['test_ROC_AUC_score_STD'][0])
    
    
    clf_list.append(clf)
 
    
              
df = pd.DataFrame(my_dict, index=clf_list)


outname = f'summary_scores_2OS_Merged.csv'

outdir = f'/home/leonardo/Scrivania/result_PA/06_03/score_ROC_AUC_optimization_using_all_HP_sets_USING_MEAN/2_classes_OS/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    
df.to_csv(fullname)


#
#a = os.path.split(name)[-1]
#a = a[12:-4]
#
#b = data['accuracy_train_mean'][0]
#
#
#my_dict = {'ACC_TRAIN_MEAN': [data['accuracy_train_mean'][0]],
#           'ACC_TRAIN_STD': [data['accuracy_train_std'][0]], 
#           'ACC_TEST_MEAN': [data['accuracy_test_mean'][0]],
#           'ACC_Test_std': [data['accuracy_test_std'][0]]}
#
#my_dict['ACC_TRAIN_MEAN'].append(3)
#
#c=pd.DataFrame(my_dict, index=[a])
#c.index.name = 'classifier'



#in caso dovessi concatenare delle colonne con dimensioni diverse conviene fare i
#dataframe di ogni colonna e poi concatenarli usando 
#df_tot = pd.concat([df_1, df_2, df_3, df_4, df_5], axis=1)

