#Cross Validation on LinearRegression for regression

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
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
import load_data_ST
import save_output
import GSCV

name_clf = 'LinearRegression'

#load data

data_train, labels_train, data_test, labels_test  = load_data_ST.function_load_data_ST()

#Scalers

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler(), None]

df = pd.DataFrame()

# Designate distributions to sample hyperparameters from 
n_features_to_test = [0.85, 0.9, 0.95]



clf = TransformedTargetRegressor(regressor=LinearRegression(),
                                     transformer=MinMaxScaler())


#LinearRegression
steps = [('scaler', StandardScaler()), ('red_dim', PCA()), ('clf', clf)]

pipeline = Pipeline(steps)

parameteres = [{'scaler':scalers_to_test, 'red_dim':[PCA(random_state=42)], 'red_dim__n_components':list(n_features_to_test)},
               {'scaler':scalers_to_test, 'red_dim':[None]}]

results = GSCV.function_GSCV(data_train, labels_train, data_test, labels_test, pipeline, parameteres)

#create folder and save

save_output.function_save_output(results, name_clf)








