#Cross Validation on RadiusNeighborsClassifier for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif
import load_data_3_class
import save_output
import nested_cv_3_classes

name_clf = 'RadiusNeighborsClassifier'


#load data and vectorize label

data, labels = load_data_3_class.function_load_data_3_class()


#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

df = pd.DataFrame()

# Designate distributions to sample hyperparameters from 
n_features_to_test = [0.85, 0.9, 0.95]
R = np.arange(0.1, 10, 0.2) 


#KNeighborsClassifier
steps = [('scaler', MinMaxScaler()), ('red_dim', PCA()), ('clf', RadiusNeighborsClassifier())]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA(random_state=42)], 'red_dim__n_components':n_features_to_test, 'clf__radius':R, 
                'clf__weights':['uniform', 'distance'], 'clf__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},
                {'scaler':scalers_to_test, 'red_dim':[SelectPercentile(f_classif, percentile=10)], 'clf__radius':R,
                'clf__weights':['uniform', 'distance'], 'clf__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},
                {'scaler':scalers_to_test, 'red_dim':[SelectPercentile(mutual_info_classif, percentile=10)], 'clf__radius':R, 
                'clf__weights':['uniform', 'distance'], 'clf__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},
               {'scaler':scalers_to_test, 'red_dim':[None], 'clf__radius':R,
                'clf__weights':['uniform', 'distance'], 'clf__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}]



results = nested_cv_3_classes.function_nested_cv_3_classes(data, labels, pipeline, parameteres)

#create folder and save

save_output.function_save_output(results, name_clf)



