#Cross Validation on AdaBoostClassifier for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

name = 'KPCA_AdaBoost'

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

encoder = LabelEncoder()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

df = pd.DataFrame()

# Designate distributions to sample hyperparameters from 
n_estimators = [10, 30, 50, 70, 100, 150, 200, 400, 600, 1000]
n_features_to_test = np.arange(1, 11)
lr = [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0]



for i in range(1, 11):

       #Train test split
       X_train, X_test, y_train, y_test = train_test_split(public_data, public_labels, test_size=0.3, 
       stratify=public_labels, random_state=i*500)

       #Vettorizzare i label
       train_labels_encoded = encoder.fit_transform(y_train)
       test_labels_encoded = encoder.transform(y_test)

       #RadiusNeighbors
       steps = [('scaler', MinMaxScaler()), ('red_dim', KernelPCA()), ('clf', AdaBoostClassifier(random_state=i*503))]

       pipeline = Pipeline(steps)

       parameteres = [{'scaler':scalers_to_test, 'red_dim':[KernelPCA(random_state=i*42)], 
                       'red_dim__n_components':n_features_to_test, 'red_dim__kernel':['linear', 'poly', 'rbf', 'sigmoid', 'cosine'],
                       'clf__base_estimator': [DecisionTreeClassifier(max_depth = j) for j in range(1,6)],                     
                       'clf__n_estimators':n_estimators, 'clf__learning_rate':lr, 'clf__algorithm':['SAMME', 'SAMME.R']}]

       grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5, n_jobs=-1, verbose=1)

       grid.fit(X_train, y_train)

       score_train = grid.score(X_train, y_train)
       score_test = grid.score(X_test, y_test)
       best_p = grid.best_params_

       bp = pd.DataFrame(best_p, index=[i])
       bp['accuracy_train'] = score_train
       bp['accuracy_test'] = score_test
       bp['random_state'] = i*500
       bp['random_state_pca'] = i*42
       bp['random_state_clf'] = i*503

       df = df.append(bp, ignore_index=True)

#df.to_csv('/home/users/ubaldi/TESI_PA/result_CV/large_space_NO_fixed_rand_state/AdaBoost_stability/best_params_AdaBoost.csv')


#insert sccuracy mean and std

acc_train_mean = df['accuracy_train'].mean()
acc_test_mean = df['accuracy_test'].mean()

acc_train_std = df['accuracy_train'].std()
acc_test_std = df['accuracy_test'].std()


df_train_acc_mean = pd.DataFrame([{'accuracy_train_mean':acc_train_mean}])
df_train_acc_std = pd.DataFrame([{'accuracy_train_std':acc_train_std}])


df_test_acc_mean = pd.DataFrame([{'accuracy_test_mean':acc_test_mean}])
df_test_acc_std = pd.DataFrame([{'accuracy_test_std':acc_test_std}])


df_tot = pd.concat([df, df_train_acc_mean, df_train_acc_std, df_test_acc_mean, df_test_acc_std], axis=1)



#create folder and save

import os

outname = f'best_params_{name}.csv'

outdir = '/home/users/ubaldi/TESI_PA/result_CV/3_classes_H/Public/large_space_change_expl_all_rand_state/AdaBoost_stability'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

df_tot.to_csv(fullname)