import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import os


#load data

train_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_training2.csv'
test_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_nostro_without_nan.csv'

df_train = pd.read_csv(train_dataset_path)
df_test = pd.read_csv(test_dataset_path)

df_train.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)
df_test.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)


df_train.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)
df_test.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)

public_data = df_train.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)
PA_data = df_test.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)

public_labels = df_train.Histology
PA_labels = df_test.Histology


#vettorizzare i label
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

tot_random_state = []
tot_train_score = []
tot_test_score = []
tot_macro_ovo = []
tot_weighted_ovo = []
tot_macro_ovr = []
tot_weighted_ovr = []


for i in range(1,31):

    #train test split 
    X_train, X_test, y_train, y_test = train_test_split(public_data, public_labels, test_size=0.3, stratify=public_labels, random_state=i*500)

    tot_random_state.append(500*i)

    #vettorizzare i label
    train_labels_encoded = encoder.fit_transform(y_train)
    test_labels_encoded = encoder.transform(y_test)

    svm = SVC(kernel='poly', probability=True)

    steps = [('clf', svm)]    

    pipeline = Pipeline(steps)

    summary = pipeline.named_steps

    pipeline.fit(X_train, train_labels_encoded)

    score_train = pipeline.score(X_train, train_labels_encoded)
    tot_train_score.append(score_train)

    score_test = pipeline.score(X_test, test_labels_encoded)
    tot_test_score.append(score_test)

    y_scores = pipeline.predict_proba(X_test)

    macro_ovo = roc_auc_score(test_labels_encoded, y_scores, average='macro',  multi_class='ovo')
    weighted_ovo = roc_auc_score(test_labels_encoded, y_scores, average='weighted',  multi_class='ovo')
    macro_ovr = roc_auc_score(test_labels_encoded, y_scores, average='macro',  multi_class='ovr')
    weighted_ovr = roc_auc_score(test_labels_encoded, y_scores, average='weighted',  multi_class='ovr')

    tot_macro_ovo.append(macro_ovo)
    tot_weighted_ovo.append(weighted_ovo)
    tot_macro_ovr.append(macro_ovr)
    tot_weighted_ovr.append(weighted_ovr)

    y_pred = pipeline.predict(X_test)
    
    report = classification_report(test_labels_encoded, y_pred, output_dict=True)
    df_r = pd.DataFrame(report)
    df_r = df_r.transpose()
    
    #create folder and save

    outname = 'report_{i}.csv'

    outdir = '/home/users/ubaldi/TESI_PA/result_score/Public/score_NOprep_NOfeatRed/report_svm_poly_NO_NO'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    

    df_r.to_csv(fullname)




# pandas can convert a list of lists to a dataframe.
# each list is a row thus after constructing the dataframe
# transpose is applied to get to the user's desired output. 
df = pd.DataFrame([tot_random_state, tot_train_score, tot_test_score, tot_macro_ovo, tot_weighted_ovo, tot_macro_ovr, tot_weighted_ovr])
df = df.transpose() 

fieldnames = ['random_state','train_accuracy','test_accuracy', 'roc_auc_score_macro_ovo', 'roc_auc_score_weighted_ovo', 
                  'roc_auc_score_macro_ovr', 'roc_auc_score_weighted_ovr']
# write the data to the specified output path: "output"/+file_name
# without adding the index of the dataframe to the output 
# and without adding a header to the output. 
# => these parameters are added to be fit the desired output. 
#df.to_csv('/home/users/ubaldi/TESI_PA/result_CV/score_svm_linear.csv', index=False, header=fieldnames)




#create folder and save

import os

outname = 'score_svm_poly_NO_NO.csv'

outdir = '/home/users/ubaldi/TESI_PA/result_score/Public/score_NOprep_NOfeatRed/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    

df.to_csv(fullname, index=False, header=fieldnames)
