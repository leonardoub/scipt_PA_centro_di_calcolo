#Cross Validation on RandomForestClassifier for regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor

name = 'RandomForestRegressor'


#load data

train_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_training2.csv'
test_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_nostro.csv'

df_train = pd.read_csv(train_dataset_path)
df_test = pd.read_csv(test_dataset_path)

df_train.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)
df_test.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)

df_train.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)
df_test.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)

public_data = df_train.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)
PA_data = df_test.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)

public_labels = df_train.Surv_time_months
#PA_labels = df_test.Surv_time_months

public_labels = public_labels.astype('float')
#PA_labels = PA_labels.astype('float')

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

df = pd.DataFrame()

# Designate distributions to sample hyperparameters from 
n_tree = np.arange(5, 120, 5)
n_features_to_test = np.arange(1, 11)


for i in range(1, 21):

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(public_data, public_labels, test_size=0.3, 
    stratify=public_labels, random_state=i*500)

    clf = TransformedTargetRegressor(regressor=RandomForestRegressor(criterion='mae'),
                                     transformer=MinMaxScaler())

    #RandomForestClassifier
    steps = [('scaler', StandardScaler()), ('red_dim', PCA()), ('clf', clf)]

    pipeline = Pipeline(steps)

    parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA()], 'red_dim__n_components':list(n_features_to_test), 'clf__regressor__n_estimators':list(n_tree), 'clf__regressor__bootstrap':[True, False]},
                   {'scaler':scalers_to_test, 'red_dim':[LinearDiscriminantAnalysis()], 'red_dim__n_components':[2], 'clf__regressor__n_estimators':list(n_tree), 'clf__regressor__bootstrap':[True, False]},
                   {'scaler':scalers_to_test, 'red_dim':[None], 'clf__regressor__n_estimators':list(n_tree), 'clf__regressor__bootstrap':[True, False]}]

    grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_absolute_error')

    grid.fit(X_train, y_train)

    score_train = grid.score(X_train, y_train)
    score_test = grid.score(X_test, y_test)
    best_p = grid.best_params_

    bp = pd.DataFrame(best_p, index=[i])
    bp['MAE_train'] = -score_train
    bp['MAE_test'] = -score_test
    bp['random_state'] = i*500

    df = df.append(bp, ignore_index=True)

#df.to_csv('/home/users/ubaldi/TESI_PA/result_CV/large_space_NO_fixed_rand_state/RandomForest_stability/best_params_RandomForest.csv')

#create folder and save

import os

outname = f'best_params_{name}_ST_regression.csv'

outdir = f'/home/users/ubaldi/TESI_PA/result_CV/regression_ST/Public/large_space_change_expl_TTS_rand_state/{name}_stability_ST_regression'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

df.to_csv(fullname)

















