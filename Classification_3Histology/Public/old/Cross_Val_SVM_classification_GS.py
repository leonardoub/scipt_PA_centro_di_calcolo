#Cross Validation on SVM for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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

#SVM

from sklearn.svm import SVC

steps = [('scaler', StandardScaler()), ('red_dim', PCA()), ('clf', SVC(kernel='linear'))]

from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps)

n_features_to_test = np.arange(1, 11)

parameteres = [{'scaler':scalers_to_test, 'red_dim':[LinearDiscriminantAnalysis()], 'red_dim__n_components':[2], 'clf__C':[0.001,0.1,10]},
               {'scaler':scalers_to_test, 'red_dim':[PCA()], 'red_dim__n_components':n_features_to_test,'clf__C':[0.001,0.1,10]}]


from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV

grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5, verbose=1, scoring='accuracy', n_jobs=-1)

print('a')

grid.fit(X_train, y_train)

print('b')

score = grid.score(X_test, y_test)
best_p = grid.best_params_
#print(f'score = {grid.score(X_test, y_test)}')
#print(grid.best_params_)

file_score = open('/home/users/ubaldi/TESI_PA/result_CV/score_lin_svm.txt', 'w')
file_score.write(f'{score}')
file_score.close()


file_best_params = open('/home/users/ubaldi/TESI_PA/result_CV/best_params_lin_svm.txt', 'w')
file_best_params.write(f'{best_p}')
file_best_params.close()




