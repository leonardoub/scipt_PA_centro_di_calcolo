#Cross Validation on RandomForestRegressor for regression

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
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
import load_data_ST
import save_output
import nested_cv_ST

name_clf = 'RandomForestRegressor'


#load data

data, labels = load_data_ST.function_load_data_ST()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

df = pd.DataFrame()

# Designate distributions to sample hyperparameters from 
n_features_to_test = [0.85, 0.9, 0.95]
n_tree = [10, 30, 50, 100]
depth = [10, 30, 50, None]

 
clf = TransformedTargetRegressor(regressor=RandomForestRegressor(criterion='mae', random_state=503),
                                     transformer=MinMaxScaler())


#RandomForestRegressor
steps = [('scaler', StandardScaler()), ('red_dim', PCA()), ('clf', clf)]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA(random_state=42)], 'red_dim__n_components':list(n_features_to_test), 
                'clf__n_estimators':list(n_tree), 'clf__max_depth':depth},
                {'scaler':scalers_to_test, 'red_dim':[None], 'clf__n_estimators':list(n_tree), 'clf__max_depth':depth}]

results = nested_cv_ST.function_nested_cv_ST(data, labels, pipeline, parameteres)

#create folder and save

save_output.function_save_output(results, name_clf)


















