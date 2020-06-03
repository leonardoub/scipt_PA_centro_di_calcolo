from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os
from sklearn.pipeline import Pipeline
import score_cv_3_classes

name = 'Adaboost'
dim_reduction = 'NONE'

#load data
import load_data_3_class 
import save_output

public_data, public_labels = load_data_3_class.function_load_data()

def create_score_csv_default_HP(scaler_, RS_outer_KF):


    n_comp_pca = None
    #whiten_ = False
    depth_= 'default'
    algorithm_ = 'default'
    lr = 'default'
    n_estimators_ = 'default'
    random_state_clf = 503
    #random_state_PCA = 42
    #random_state_outer_kf = RS_outer_KF

    dict_best_params = {'SCALER':[scaler_], 'PCA__n_components':[n_comp_pca],  
                        'CLF__algorithm':[algorithm_], 'CLF__lr':[lr], 'CLF__n_estimators':[n_estimators_],
                        'CLF__base_estimator_DTC_max_depth':[depth_],
                        'CLF__random_state':[random_state_clf], 'random_state_outer_kf':[RS_outer_KF]}


    df_best_params = pd.DataFrame.from_dict(dict_best_params)

    #implmentation of steps
    scaler=scaler_
    #pca = PCA(n_components=n_comp_pca, whiten=whiten_, random_state=random_state_PCA)
    clf = AdaBoostClassifier(random_state=random_state_clf)

    steps = [('scaler', scaler), ('clf', clf)]    
    pipeline = Pipeline(steps)

    df_score_value, df_mean_std = score_cv.function_score_cv(public_data, public_labels, pipeline, RS_outer_KF)
    df_tot = pd.concat([df_best_params, df_score_value, df_mean_std], axis=1, ignore_index=False)


    return df_tot


for j in range(1,2):

    df_MMS = create_score_csv_default_HP(MinMaxScaler(), 2*j)
    save_output.function_save_output(df_MMS, 'MMS', name, 2*j)

    df_STDS = create_score_csv_default_HP(StandardScaler(), 2*j)
    save_output.function_save_output(df_STDS, 'STDS', name, 2*j)

    df_RBT = create_score_csv_default_HP(RobustScaler(), 2*j)
    save_output.function_save_output(df_RBT, 'RBT', name, 2*j)

    df_NONE = create_score_csv_default_HP(None, 2*j)
    save_output.function_save_output(df_NONE, 'NONE', name, 2*j)

