#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:42:00 2020

@author: leonardo
"""

import pandas as pd

path = '/home/leonardo/Scrivania/result_PA/06_03/result_CV_bis/2_classes_OS/Merged_OS_1_2/*/*.csv'



import glob
for name in glob.glob(path):
    print(name)
    data = pd.read_csv(name) 
    
    
    
    acc_train_mean = data['train_accuracy_scores'].mean()
    acc_train_std = data['train_accuracy_scores'].std()

    acc_test_mean = data['outer_loop_accuracy_scores'].mean()
    acc_test_std = data['outer_loop_accuracy_scores'].std()



    bal_acc_train_mean = data['train_balanced_accuracy_scores'].mean()
    bal_acc_train_std = data['train_balanced_accuracy_scores'].std()

    bal_acc_test_mean = data['outer_loop_balanced_accuracy_scores'].mean()
    bal_acc_test_std = data['outer_loop_balanced_accuracy_scores'].std()
    
    
        
    ROC_AUC_train_mean = data['train_roc_auc_scores_predict_proba'].mean()
    ROC_AUC_train_std = data['train_roc_auc_scores_predict_proba'].std()
    
    ROC_AUC_test_mean = data['outer_loop_roc_auc_scores_predict_proba'].mean()
    ROC_AUC_test_std = data['outer_loop_roc_auc_scores_predict_proba'].std()

    
    
    df_train_acc_mean = pd.DataFrame([{'train_accuracy_MEAN':acc_train_mean}])
    df_train_acc_std = pd.DataFrame([{'train_accuracy_STD':acc_train_std}])
    
    df_test_acc_mean = pd.DataFrame([{'test_accuracy_MEAN':acc_test_mean}])
    df_test_acc_std = pd.DataFrame([{'test_accuracy_STD':acc_test_std}])


    df_train_bal_acc_mean = pd.DataFrame([{'train_balanced_accuracy_MEAN':bal_acc_train_mean}])
    df_train_bal_acc_std = pd.DataFrame([{'train_balanced_accuracy_STD':bal_acc_train_std}])

    df_test_bal_acc_mean = pd.DataFrame([{'test_balanced_accuracy_MEAN':bal_acc_test_mean}])
    df_test_bal_acc_std = pd.DataFrame([{'test_balanced_accuracy_STD':bal_acc_test_std}])

    
    df_train_ROC_AUC_mean = pd.DataFrame([{'train_ROC_AUC_score_MEAN':ROC_AUC_train_mean}])
    df_train_ROC_AUC_std = pd.DataFrame([{'train_ROC_AUC_score_STD':ROC_AUC_train_std}])
    
    df_test_ROC_AUC_mean = pd.DataFrame([{'test_ROC_AUC_score_MEAN':ROC_AUC_test_mean}])
    df_test_ROC_AUC_std = pd.DataFrame([{'test_ROC_AUC_score_STD':ROC_AUC_test_std}])


    df = pd.concat([data, df_train_acc_mean, df_train_acc_std, df_test_acc_mean, df_test_acc_std,
                    df_train_bal_acc_mean, df_train_bal_acc_std, df_test_bal_acc_mean, df_test_bal_acc_std,
                    df_train_ROC_AUC_mean, df_train_ROC_AUC_std, df_test_ROC_AUC_mean, df_test_ROC_AUC_std], axis=1)

    df.to_csv(name, index=False)


#data = pd.read_csv(name) 

#acc_train_mean = data['accuracy_train'].mean()

#data['accuracy_train'] = acc_train_mean




