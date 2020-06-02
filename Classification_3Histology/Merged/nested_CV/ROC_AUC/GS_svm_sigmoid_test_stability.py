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
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif
import load_data_3_class
import save_output
import nested_cv_3_classes

name_clf = 'SVM_sigmoid'


#load data

data, labels = load_data_3_class.function_load_data_3_class()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler()]

df = pd.DataFrame()

# Designate distributions to sample hyperparameters from 
C_range = np.power(2, np.arange(-10, 9, dtype=float))
gamma_range = np.power(2, np.arange(-10, 9, dtype=float))
n_features_to_test = [0.85, 0.9, 0.95]


#SVM
steps = [('scaler', StandardScaler()), ('red_dim', PCA()), ('clf', SVC(kernel='sigmoid', probability=True))]

pipeline = Pipeline(steps)


parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA(random_state=42)], 'red_dim__n_components':list(n_features_to_test),
              'clf__C': list(C_range), 'clf__gamma':['auto', 'scale']+list(gamma_range)},
              {'scaler':scalers_to_test, 'red_dim':[SelectPercentile(f_classif, percentile=10)],
              'clf__C': list(C_range), 'clf__gamma':['auto', 'scale']+list(gamma_range)},
              {'scaler':scalers_to_test, 'red_dim':[SelectPercentile(mutual_info_classif, percentile=10)],
              'clf__C': list(C_range), 'clf__gamma':['auto', 'scale']+list(gamma_range)},
              {'scaler':scalers_to_test, 'red_dim':[None],
              'clf__C': list(C_range), 'clf__gamma':['auto', 'scale']+list(gamma_range)}]

results = nested_cv_3_classes.function_nested_cv_3_classes(data, labels, pipeline, parameteres)

#create folder and save

save_output.function_save_output(results, name_clf)



