#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:55:48 2020

@author: leonardo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:42:00 2020

@author: leonardo
"""

import pandas as pd

path = '/home/leonardo/Scrivania/result_PA/06_03/result_CV_bis/2_classes_OS/*/*/*.csv'

import glob
for name in glob.glob(path):
    #print(name)
    data = pd.read_csv(name) 


    acc_train_mean = data['train_accuracy_scores'].mean()
    acc_test_mean = data['outer_loop_accuracy_scores'].mean()
    
    acc_train_std = data['train_accuracy_scores'].std()
    acc_test_std = data['outer_loop_accuracy_scores'].std()

###################################################

    bal_acc_train_mean = data['train_balanced_accuracy_scores'].mean()
    bal_acc_test_mean = data['outer_loop_balanced_accuracy_scores'].mean()
    
    bal_acc_train_std = data['train_balanced_accuracy_scores'].std()
    bal_acc_test_std = data['outer_loop_balanced_accuracy_scores'].std()

####################################################    
    
    roc_auc_train_mean = data['train_roc_auc_scores_predict_proba'].mean()
    roc_auc_test_mean = data['test_loop_roc_auc_scores_predict_proba'].mean()
    
    roc_auc_train_std = data['train_roc_auc_scores_predict_proba'].std()
    roc_auc_test_std = data['test_loop_roc_auc_scores_predict_proba'].std()



    df_train_acc_mean = pd.DataFrame([{'accuracy_train_mean':acc_train_mean}])
    df_train_acc_std = pd.DataFrame([{'accuracy_train_std':acc_train_std}])
    
    df_test_acc_mean = pd.DataFrame([{'accuracy_test_mean':acc_test_mean}])
    df_test_acc_std = pd.DataFrame([{'accuracy_test_std':acc_test_std}])


    df_train_bal_acc_mean = pd.DataFrame([{'bal_accuracy_train_mean':bal_acc_train_mean}])
    df_train_bal_acc_std = pd.DataFrame([{'bal_accuracy_train_std':bal_acc_train_std}])
    
    df_test_bal_acc_mean = pd.DataFrame([{'bal_accuracy_test_mean':bal_acc_test_mean}])
    df_test_bal_acc_std = pd.DataFrame([{'bal_accuracy_test_std':bal_acc_test_std}])
    
    
    df_train_roc_auc_mean = pd.DataFrame([{'roc_auc_train_mean':roc_auc_train_mean}])
    df_train_roc_auc_std = pd.DataFrame([{'roc_auc_train_std':roc_auc_train_std}])
    
    df_test_roc_auc_mean = pd.DataFrame([{'roc_auc_test_mean':roc_auc_test_mean}])
    df_test_roc_auc_std = pd.DataFrame([{'roc_auc_test_std':roc_auc_test_std}])


    tot_data = [data, df_train_acc_mean, df_train_acc_std, df_test_acc_mean, df_test_acc_std,
                df_train_bal_acc_mean, df_train_bal_acc_std, df_test_bal_acc_mean, df_test_bal_acc_std,
                df_train_roc_auc_mean, df_train_roc_auc_std, df_test_roc_auc_mean, df_test_roc_auc_std]




    df = pd.concat(tot_data, axis=1)

    df.to_csv(name, index=False)




#data = pd.read_csv(name) 

#acc_train_mean = data['accuracy_train'].mean()

#data['accuracy_train'] = acc_train_mean




