#Cross Validation on SVM for regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression

name = 'svmR_poly_2'

#load data

train_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_training2_without_nan_OS.csv'
test_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_nostro.csv'

df_train = pd.read_csv(train_dataset_path)
df_test = pd.read_csv(test_dataset_path)

df_train.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)
df_test.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)

df_train.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)
df_test.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)

df_train.loc[df_train['Overall_Stage'] == 'I', 'Overall_Stage'] = 1
df_train.loc[df_train['Overall_Stage'] == 'II', 'Overall_Stage'] = 2
df_train.loc[df_train['Overall_Stage'] == 'IIIa', 'Overall_Stage'] = 3
df_train.loc[df_train['Overall_Stage'] == 'IIIb', 'Overall_Stage'] = 4

public_data = df_train.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)
PA_data = df_test.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)

public_labels = df_train.Overall_Stage
#PA_labels = df_test.Overall_Stage

public_labels = public_labels.astype('int')
#PA_labels = PA_labels.astype('int')

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

df = pd.DataFrame()

# Designate distributions to sample hyperparameters from 
C_range = np.power(2, np.arange(-10, 11, dtype=float))
gamma_range = np.power(2, np.arange(-10, 11, dtype=float))


for i in range(1, 21):

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(public_data, public_labels, test_size=0.3, 
    stratify=public_labels, random_state=i*500)

    clf = TransformedTargetRegressor(regressor=SVR(kernel='poly'),
                                     transformer=MinMaxScaler())


    #LinearRegression
    steps = [('scaler', StandardScaler()), ('red_dim', PCA()), ('clf', clf)]

    pipeline = Pipeline(steps)
    n_features_to_test = np.arange(1, 11)

    parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA()], 'red_dim__n_components':list(n_features_to_test),
                    'clf__regressor__C': list(C_range), 'clf__regressor__gamma':['auto', 'scale']+list(gamma_range), 'clf__regressor__degree':[2]},
                    {'scaler':scalers_to_test, 'red_dim':[LinearDiscriminantAnalysis()], 'red_dim__n_components':[2],
                    'clf__regressor__C': list(C_range), 'clf__regressor__gamma':['auto', 'scale']+list(gamma_range), 'clf__regressor__degree':[2]},
                    {'scaler':scalers_to_test, 'red_dim':[None],
                    'clf__regressor__C': list(C_range), 'clf__regressor__gamma':['auto', 'scale']+list(gamma_range), 'clf__regressor__degree':[2]}]

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

outname = f'best_params_{name}_OS_regression.csv'

outdir = f'/home/users/ubaldi/TESI_PA/result_CV/regression_OS/Public/large_space_change_expl_TTS_rand_state/{name}_stability_OS_regression'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

df.to_csv(fullname)

