#Cross Validation on RandomForestClassifier for classification

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
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import load_data_2_class
import save_output
import GSCV

name_clf = 'RandomForestClassifier'


#load data

X_train, y_train, X_test, y_test = load_data_2_class.function_load_data_2_class()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

df = pd.DataFrame()

# Designate distributions to sample hyperparameters from 
n_features_to_test = [0.85, 0.9, 0.95]
n_tree = [10, 30, 50, 70, 100]
depth = [10, 25, 50, 75, None]




#RandomForestClassifier
steps = [('scaler', StandardScaler()), ('red_dim', PCA()), ('clf', RandomForestClassifier())]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA(random_state=42)], 'red_dim__n_components':list(n_features_to_test), 
                'clf__n_estimators':list(n_tree), 'clf__max_depth':depth, 'clf__class_weight':[None, 'balanced']},
                {'scaler':scalers_to_test, 'red_dim':[SelectPercentile(f_classif, percentile=10)],
                'clf__n_estimators':list(n_tree), 'clf__max_depth':depth, 'clf__class_weight':[None, 'balanced']},
                {'scaler':scalers_to_test, 'red_dim':[SelectPercentile(mutual_info_classif, percentile=10)],
                'clf__n_estimators':list(n_tree), 'clf__max_depth':depth, 'clf__class_weight':[None, 'balanced']},
                {'scaler':scalers_to_test, 'red_dim':[None], 'clf__n_estimators':list(n_tree), 'clf__max_depth':depth,
                'clf__class_weight':[None, 'balanced']}]


results = GSCV.function_GSCV(X_train, y_train, X_test, y_test, pipeline, parameteres)

#create folder and save

save_output.function_save_output(results, name_clf)


















