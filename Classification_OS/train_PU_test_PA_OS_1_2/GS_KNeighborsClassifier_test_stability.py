#Cross Validation on KNeighborsClassifier for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif
import load_data_2_class
import save_output
import GSCV

name_clf = 'KNeighborsClassifier'


#load data

X_train, y_train, X_test, y_test = load_data_2_class.function_load_data_2_class()


#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

df = pd.DataFrame()

# Designate distributions to sample hyperparameters from 
n_features_to_test = [0.85, 0.9, 0.95]
k = np.arange(1,11)


#KNeighborsClassifier
steps = [('scaler', MinMaxScaler()), ('red_dim', PCA()), ('clf', KNeighborsClassifier())]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA(random_state=42)], 'red_dim__n_components':n_features_to_test, 'clf__n_neighbors':k, 
                'clf__weights':['uniform', 'distance'], 'clf__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},
                {'scaler':scalers_to_test, 'red_dim':[SelectPercentile(f_classif, percentile=10)], 'clf__n_neighbors':k, 
                'clf__weights':['uniform', 'distance'], 'clf__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},
                {'scaler':scalers_to_test, 'red_dim':[SelectPercentile(mutual_info_classif, percentile=10)], 'clf__n_neighbors':k, 
                'clf__weights':['uniform', 'distance'], 'clf__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},
               {'scaler':scalers_to_test, 'red_dim':[None], 'clf__n_neighbors':k, 
                'clf__weights':['uniform', 'distance'], 'clf__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}]



results = GSCV.function_GSCV(X_train, y_train, X_test, y_test, pipeline, parameteres)

#create folder and save

save_output.function_save_output(results, name_clf)

