#Cross Validation on SVM for classification

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

#load data

train_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_training2.csv'
test_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_nostro_without_nan.csv'

df_train = pd.read_csv(train_dataset_path)
df_test = pd.read_csv(test_dataset_path)

df_train.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)
df_test.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)

df_train.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)
df_test.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)


#select histologies
df_train_LS = df_train[df_train['Histology'] != 'adenocarcinoma']
df_test_LS = df_test[df_test['Histology'] != 'adenocarcinoma']


public_data = df_train_LS.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)
PA_data = df_test_LS.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)

public_labels = df_train_LS.Histology
PA_labels = df_test_LS.Histology

encoder = LabelEncoder()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

df = pd.DataFrame()

# Designate distributions to sample hyperparameters from 
C_range = np.power(2, np.arange(-10, 11, dtype=float))
gamma_range = np.power(2, np.arange(-10, 11, dtype=float))
n_features_to_test = np.arange(2,10)


for i in range(1, 21):

       #Train test split
       X_train, X_test, y_train, y_test = train_test_split(public_data, public_labels, test_size=0.3, 
       stratify=public_labels, random_state=i*500)

       #Vettorizzare i label
       train_labels_encoded = encoder.fit_transform(y_train)
       test_labels_encoded = encoder.transform(y_test)

       #SVM
       steps = [('scaler', StandardScaler()), ('red_dim', PCA()), ('clf', SVC(kernel='sigmoid'))]

       pipeline = Pipeline(steps)

       n_features_to_test = np.arange(1, 11)

       parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA()], 'red_dim__n_components':n_features_to_test,
                     'clf__C': list(C_range), 'clf__gamma':['auto', 'scale']+list(gamma_range)},
                     {'scaler':scalers_to_test, 'red_dim':[None],
                     'clf__C': list(C_range), 'clf__gamma':['auto', 'scale']+list(gamma_range)}]


       grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5, n_jobs=-1, verbose=1)

       grid.fit(X_train, y_train)
       
       score_train = grid.score(X_train, y_train)
       score_test = grid.score(X_test, y_test)
       best_p = grid.best_params_

       bp = pd.DataFrame(best_p, index=[i])
       bp['accuracy_train'] = score_train
       bp['accuracy_test'] = score_test
       bp['random_state'] = i*500

       df = df.append(bp, ignore_index=True)

#df.to_csv('/home/users/ubaldi/TESI_PA/result_CV/large_space_NO_fixed_rand_state/sigmoid_svm_stability/best_params_svm_sigmoid.csv')



import os

outname = 'best_params_svm_sigmoid_2_classes.csv'

outdir = '/home/users/ubaldi/TESI_PA/result_CV/2_classes_H/Public/large_space_change_expl_TTS_rand_state/sigmoid_svm_stability'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

df.to_csv(fullname)

