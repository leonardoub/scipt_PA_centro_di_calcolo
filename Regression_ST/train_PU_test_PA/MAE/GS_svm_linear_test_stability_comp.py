#Cross Validation on SVM for classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.compose import TransformedTargetRegressor
import load_data_ST
import save_output
import GSCV

name_clf = 'SVMR_linear_comp'


#load data

data_train, labels_train, data_test, labels_test  = load_data_ST.function_load_data_ST()

#Scalers
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler()]

df = pd.DataFrame()

#Designate distributions to sample hyperparameters from 
C_range = np.power(2, np.arange(-10, 8, dtype=float))
n_features_to_test = [0.85, 0.9, 0.95]


clf = TransformedTargetRegressor(regressor=SVR(kernel='linear'),
                                  transformer=MinMaxScaler())


#SVM
steps = [('scaler', StandardScaler()), ('red_dim', PCA()), ('clf',clf)]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA(random_state=42)], 'red_dim__n_components':list(n_features_to_test), 'clf__regressor__C':list(C_range)},
                {'scaler':scalers_to_test, 'red_dim':[None], 'clf__regressor__C':list(C_range)}]


results = GSCV.function_GSCV(data_train, labels_train, data_test, labels_test, pipeline, parameteres)

#create folder and save

save_output.function_save_output(results, name_clf)



