#Cross Validation on SVM for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
import load_data
import save_output
import nested_cv

name = 'svm_poly'
#dim_reduction = 'PCA'

#load data

public_data, public_labels = load_data.function_load_data()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]


# Designate distributions to sample hyperparameters from 
C_range = np.power(2, np.arange(-10, 11, dtype=float))
gamma_range = np.power(2, np.arange(-10, 11, dtype=float))
n_features_to_test = [0.85, 0.9, 0.95]


#SVM
steps = [('scaler', MinMaxScaler()), ('red_dim', PCA()), ('clf', SVC(kernel='poly', max_iter=1000, probability=True, random_state=503))]

pipeline = Pipeline(steps)


parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA(random_state=42)], 'red_dim__n_components':list(n_features_to_test), 
              'clf__C': list(C_range), 'clf__gamma':['auto', 'scale']+list(gamma_range), 'clf__degree':[2, 3], 'clf__class_weight':[None, 'balanced']}]


for j in range(1,2):
    results, best_estimators_dict = nested_cv.function_nested_cv(public_data, public_labels, pipeline, parameteres, j*2)

    #create folder and save

    save_output.function_save_output(results, name, j*2)

