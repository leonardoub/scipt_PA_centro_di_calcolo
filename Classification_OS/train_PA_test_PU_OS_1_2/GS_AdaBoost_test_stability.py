#Cross Validation on AdaBoostClassifier for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
import load_data_2_class
import save_output
import GSCV

name_clf = 'AdaBoost'


#load data and vectorize label

X_train, y_train, X_test, y_test = load_data_2_class.function_load_data_2_class()


#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

df = pd.DataFrame()

# Designate distributions to sample hyperparameters from 
n_features_to_test = [0.85, 0.9, 0.95]
n_estimators = [10, 30, 50, 70, 100, 150]
lr = [0.001, 0.01, 0.1, 0.50, 1.0]



for i in range(1, 11):

       inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i*42)


       #RadiusNeighbors
       steps = [('scaler', MinMaxScaler()), ('red_dim', PCA()), ('clf', AdaBoostClassifier())]

       pipeline = Pipeline(steps)


       parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA()], 'red_dim__n_components':n_features_to_test, 
                       'clf__n_estimators':n_estimators, 'clf__learning_rate':lr, 'clf__algorithm':['SAMME', 'SAMME.R']},
                       {'scaler':scalers_to_test, 'red_dim':[None], 
                       'clf__n_estimators':n_estimators, 'clf__learning_rate':lr, 'clf__algorithm':['SAMME', 'SAMME.R']}]

       grid = GridSearchCV(pipeline, param_grid=parameteres, cv=inner_kf, n_jobs=-1, verbose=1)

       grid.fit(X_train, y_train)

       score_train = grid.score(X_train, y_train)
       score_test = grid.score(X_test, y_test)
       best_p = grid.best_params_

       bp = pd.DataFrame(best_p, index=[i])
       bp['accuracy_train'] = score_train
       bp['accuracy_test'] = score_test
       bp['random_state_k_fold'] = i*42


       df = df.append(bp, ignore_index=True)



#create folder and save

save_output.function_save_output(df, name_clf)

