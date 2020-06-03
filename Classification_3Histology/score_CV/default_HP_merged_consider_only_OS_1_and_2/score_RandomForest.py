import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import score_cv

name = 'RandomForest'
dim_reduction = 'NONE'


#load data
import load_data 
import save_output

public_data, public_labels = load_data.function_load_data()

def create_score_csv_default_HP(scaler_, RS_outer_KF):


    n_comp_pca = None
    #whiten_ = False
    n_estimators_ = 'default'
    criterion_ = 'default'
    bootstrap_ = 'default'
    max_depth_ = 'default'
    min_samples_split_ = 'default'
    min_samples_leaf_ = 'default'
    class_weight_ = 'default'
    random_state_clf = 503
    #random_state_PCA = 42
    #random_state_outer_kf = RS_outer_KF

    dict_best_params = {'SCALER':[scaler_], 'PCA__n_components':[n_comp_pca], 
                        'CLF__n_estimators':[n_estimators_], 'CLF__criterion':[criterion_], 'CLF__bootstrap':[bootstrap_],
                        'CLF__max_depth':[max_depth_], 'CLF__min_samples_split':[min_samples_split_], 
                        'CLF__min_samples_leaf':[min_samples_leaf_], 'CLF__class_weight':[class_weight_],
                        'CLF__random_state':[random_state_clf], 'random_state_outer_kf':[RS_outer_KF]}

    df_best_params = pd.DataFrame.from_dict(dict_best_params)

    #implmentation of steps
    scaler=scaler_
    #pca = PCA(n_components=n_comp_pca, whiten=whiten_, random_state=random_state_PCA)
    clf = RandomForestClassifier(random_state=random_state_clf)


    steps = [('scaler', scaler), ('clf', clf)]    
    pipeline = Pipeline(steps)


    df_score_value, df_mean_std = score_cv.function_score_cv(public_data, public_labels, pipeline, RS_outer_KF)
    df_tot=pd.concat([df_best_params, df_score_value, df_mean_std], axis=1, ignore_index=False)


    return df_tot


for j in range(1,6):

    df_MMS = create_score_csv_default_HP(MinMaxScaler(), 2*j)
    save_output.function_save_output(df_MMS, 'MMS', name, 2*j)

    df_STDS = create_score_csv_default_HP(StandardScaler(), 2*j)
    save_output.function_save_output(df_STDS, 'STDS', name, 2*j)

    df_RBT = create_score_csv_default_HP(RobustScaler(), 2*j)
    save_output.function_save_output(df_RBT, 'RBT', name, 2*j)

    df_NONE = create_score_csv_default_HP(None, 2*j)
    save_output.function_save_output(df_NONE, 'NONE', name, 2*j)