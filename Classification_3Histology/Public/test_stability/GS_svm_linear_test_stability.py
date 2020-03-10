#Cross Validation on SVM for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


#load data

train_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_training2.csv'
test_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_nostro_without_nan.csv'

df_train = pd.read_csv(train_dataset_path)
df_test = pd.read_csv(test_dataset_path)

df_train.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)
df_test.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)


df_train.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)
df_test.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)

public_data = df_train.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)
PA_data = df_test.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)

public_labels = df_train.Histology
PA_labels = df_test.Histology

#Train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(public_data, public_labels, test_size=0.3, stratify=public_labels, random_state=1)

#Vettorizzare i label

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train_labels_encoded = encoder.fit_transform(y_train)
test_labels_encoded = encoder.transform(y_test)


#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
scalers_to_test = [StandardScaler(), RobustScaler()]


# Designate distributions to sample hyperparameters from 
C_range = np.array([9.78736006e+00, 2.23814334e+01, 1.00000000e-04, 1.00000000e-04,
       1.74371223e+01, 1.00000000e-04, 2.96832303e-01, 1.06931597e+01,
       8.90706391e+00, 1.75488618e+01, 1.49564414e+01, 1.06939267e+01,
       1.00000000e-04, 7.94862668e+00, 3.14271995e+00, 1.00000000e-04,
       1.41729905e+01, 8.07236535e+00, 4.54900806e-01, 1.00000000e-04,
       1.00000000e-04, 1.99524074e+00, 4.68439119e+00, 1.00000000e-04,
       1.16220405e+01, 1.00000000e-04, 1.00000000e-04, 1.03972709e+01,
       1.00000000e-04, 1.00000000e-04, 1.00000000e-04, 1.00000000e-04,
       1.25523737e+01, 1.00000000e-04, 1.66095249e+01, 8.07308186e+00,
       1.00000000e-04, 1.00000000e-04, 1.00000000e-04, 1.00000000e-04,
       2.08711336e+01, 1.64441230e+00, 1.15020554e+01, 1.00000000e-04,
       1.81035130e+00, 1.17786194e+01, 1.00000000e-04, 1.03111446e+01,
       1.00000000e-04, 1.00000000e-04])


# Check that gamma>0 and C>0 
C_range[C_range < 0] = 0.0001


#SVM

for i in range(1,6):

    steps = [('scaler', StandardScaler()), ('red_dim', PCA()), ('clf', SVC(kernel='linear'))]

    pipeline = Pipeline(steps)

    n_features_to_test = np.arange(1, 11)

    parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA()], 'red_dim__n_components':n_features_to_test,'clf__C':C_range}]

    from sklearn.model_selection import GridSearchCV 
    from sklearn.model_selection import RandomizedSearchCV

    grid = GridSearchCV(pipeline, param_grid=parameteres, cv=3, n_jobs=-1, verbose=1)

    grid.fit(X_train, y_train)

    score = grid.score(X_test, y_test)
    best_p = grid.best_params_


    file_best_params = open(f'/home/users/ubaldi/TESI_PA/result_CV/GS_lin_svm_stability/best_params_{i}_acc_{score}.txt', 'w')
    file_best_params.write(f'{best_p}')
    file_best_params.close()




