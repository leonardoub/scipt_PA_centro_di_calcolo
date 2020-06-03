from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def function_score_cv(X_train, y_train, X_test, y_test, pipel):

    tot_train_acc = []
    tot_test_acc = []
    tot_train_bal_acc = []
    tot_test_bal_acc = []
    tot_train_ROC_AUC = []
    tot_test_ROC_AUC = []

       
    pipel.fit(X_train, y_train)
    

    pred_train = pipel.predict(X_train)       
    pred_test = pipel.predict(X_test)
    

    pred_proba_train = pipel.predict_proba(X_train)[:, 1]        
    pred_proba_test = pipel.predict_proba(X_test)[:, 1]

    #compute scoring

    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)

    train_bal_acc = balanced_accuracy_score(y_train, pred_train)
    test_bal_acc = balanced_accuracy_score(y_test, pred_test)

    train_ROC_AUC = roc_auc_score(y_train, pred_proba_train)
    test_ROC_AUC = roc_auc_score(y_test, pred_proba_test)

    
    # Appending the "winning" hyper parameters and their associated accuracy score
    

    tot_train_acc.append(train_acc)
    tot_test_acc.append(test_acc)

    tot_train_bal_acc.append(train_bal_acc)
    tot_test_bal_acc.append(test_bal_acc)

    tot_train_ROC_AUC.append(train_ROC_AUC)
    tot_test_ROC_AUC.append(test_ROC_AUC)


    #mean value and std

    mean_train_acc = np.mean(tot_train_acc)
    std_train_acc = np.std(tot_train_acc)
    mean_test_acc = np.mean(tot_test_acc)
    std_test_acc = np.std(tot_test_acc)

    
    mean_train_bal_acc = np.mean(tot_train_bal_acc)
    std_train_bal_acc = np.std(tot_train_bal_acc)
    mean_test_bal_acc = np.mean(tot_test_bal_acc)
    std_test_bal_acc = np.std(tot_test_bal_acc) 
    

    mean_train_ROC_AUC = np.mean(tot_train_ROC_AUC)
    std_train_ROC_AUC = np.std(tot_train_ROC_AUC)
    mean_test_ROC_AUC = np.mean(tot_test_ROC_AUC)
    std_test_ROC_AUC = np.std(tot_test_ROC_AUC)


    score_value_dict = {'train_accuracy':tot_train_acc, 'test_accuracy': tot_test_acc, 
    'train_balanced_accuracy':tot_train_bal_acc, 'test_balanced_accuracy':tot_test_bal_acc, 
    'train_ROC_AUC_score':tot_train_ROC_AUC, 'test_ROC_AUC_score':tot_test_ROC_AUC}

    df_score_value = pd.DataFrame.from_dict(score_value_dict)

    mean_std_dict = {'train_accuracy_MEAN':[mean_train_acc], 'train_accuracy_STD':[std_train_acc],
        'test_accuracy_MEAN':[mean_test_acc], 'test_accuracy_STD':[std_test_acc],
        'train_balanced_accuracy_MEAN':[mean_train_bal_acc], 'train_balanced_accuracy_STD':[std_train_bal_acc],
        'test_balanced_accuracy_MEAN':[mean_test_bal_acc], 'test_balanced_accuracy_STD':[std_test_bal_acc],
        'train_ROC_AUC_score_MEAN':[mean_train_ROC_AUC], 'train_ROC_AUC_score_STD':[std_train_ROC_AUC],
        'test_ROC_AUC_score_MEAN':[mean_test_ROC_AUC], 'test_ROC_AUC_score_STD':[std_test_ROC_AUC]}

    df_mean_std = pd.DataFrame.from_dict(mean_std_dict)

#        fieldnames = ['train_accuracy', 'train_accuracy_MEAN', 'train_accuracy_STD',
#           'test_accuracy', 'test_accuracy_MEAN', 'test_accuracy_STD',
#            'train_balanced_accuracy', 'train_balanced_accuracy_MEAN', 'train_balanced_accuracy_STD',
#            'test_balanced_accuracy', 'test_balanced_accuracy_MEAN', 'test_balanced_accuracy_STD',
#            'train_ROC_AUC_score', 'train_ROC_AUC_score_MEAN', 'train_ROC_AUC_score_STD',
#            'test_ROC_AUC_score', 'test_ROC_AUC_score_MEAN', 'test_ROC_AUC_score_STD']   

#        df = pd.DataFrame([tot_train_acc, [mean_train_acc], [std_train_acc], 
#                    tot_test_acc, [mean_test_acc], [std_test_acc], 
#                    tot_train_bal_acc, [mean_train_bal_acc], [std_train_bal_acc], 
#                    tot_test_bal_acc, [mean_test_bal_acc], [std_test_bal_acc],
#                    tot_train_ROC_AUC, [mean_train_ROC_AUC], [std_train_ROC_AUC],
#                    tot_test_ROC_AUC, [mean_test_ROC_AUC], [std_test_ROC_AUC]], columns=fieldnames)
#        df = df.transpose() 


    return df_score_value, df_mean_std 





