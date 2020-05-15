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
import load_data_2_class
import save_output
import GSCV

name_clf = 'SVM_linear_MMS'


#load data

X_train, y_train, X_test, y_test = load_data_2_class.function_load_data_2_class()

#Scalers
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [RobustScaler(), MinMaxScaler()]

df = pd.DataFrame()

#Designate distributions to sample hyperparameters from 
C_range = np.power(2, np.arange(-10, 11, dtype=float))
n_features_to_test = [0.85, 0.9, 0.95]




#SVM
steps = [('scaler', MinMaxScaler()), ('red_dim', PCA()), ('clf', SVC(kernel='linear', probability=True))]

pipeline = Pipeline(steps)

parameteres = [{'scaler':[MinMaxScaler()], 'red_dim':[PCA()], 'red_dim__n_components':list(n_features_to_test), 
                'clf__C':list(C_range), 'clf__class_weight':[None, 'balanced']},
              {'scaler':[MinMaxScaler()], 'red_dim':[None], 'clf__C':list(C_range), 'clf__class_weight':[None, 'balanced']}]


results = GSCV.function_GSCV(X_train, y_train, X_test, y_test, pipeline, parameteres)

#create folder and save

save_output.function_save_output(results, name_clf)



