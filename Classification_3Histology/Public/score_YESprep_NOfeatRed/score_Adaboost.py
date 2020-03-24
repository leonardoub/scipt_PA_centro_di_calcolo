import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

name = 'Adaboost'
folder = 'score_YESprep_NOfeatRed'

#load data

train_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_training2.csv'
test_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_nostro_without_nan.csv'

df_train = pd.read_csv(train_dataset_path)
df_test = pd.read_csv(test_dataset_path)

df_train.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)
df_test.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)


df_train.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)
df_test.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)

public_data = df_train.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event', 'Overall_Stage'], axis=1)
PA_data = df_test.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event', 'Overall_Stage'], axis=1)

public_labels = df_train.Histology
PA_labels = df_test.Histology


#vettorizzare i label
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

def create_csv_score_YES_NO(scaler_, abbr_scaler):

    #tot_random_state = []
    tot_train_score = []
    tot_test_score = []
    #tot_macro_ovo = []
    #tot_weighted_ovo = []
    #tot_macro_ovr = []
    tot_weighted_ovr = []

    n_comp_pca = 3
    algorithm_ = 'SAMME.R'
    lr = 0.2
    n_estimators_ = 60

    for i in range(1,31):

        #train test split 
        X_train, X_test, y_train, y_test = train_test_split(public_data, public_labels, test_size=0.3, stratify=public_labels)

        #tot_random_state.append(500*i)

        #vettorizzare i label
        train_labels_encoded = encoder.fit_transform(y_train)
        test_labels_encoded = encoder.transform(y_test)

        scaler = scaler_
        pca = PCA(n_components=n_comp_pca)
        clf = AdaBoostClassifier()

        steps = [('scaler', scaler), ('red_dim', None), ('clf', clf)]    

        pipeline = Pipeline(steps)

        summary = pipeline.named_steps

        pipeline.fit(X_train, train_labels_encoded)

        score_train = pipeline.score(X_train, train_labels_encoded)
        tot_train_score.append(score_train)

        score_test = pipeline.score(X_test, test_labels_encoded)
        tot_test_score.append(score_test)

        y_scores = pipeline.predict_proba(X_test)

        #macro_ovo = roc_auc_score(test_labels_encoded, y_scores, average='macro',  multi_class='ovo')
        #weighted_ovo = roc_auc_score(test_labels_encoded, y_scores, average='weighted',  multi_class='ovo')
        #macro_ovr = roc_auc_score(test_labels_encoded, y_scores, average='macro',  multi_class='ovr')
        weighted_ovr = roc_auc_score(test_labels_encoded, y_scores, average='weighted',  multi_class='ovr')

        #tot_macro_ovo.append(macro_ovo)
        #tot_weighted_ovo.append(weighted_ovo)
        #tot_macro_ovr.append(macro_ovr)
        tot_weighted_ovr.append(weighted_ovr)

        y_pred = pipeline.predict(X_test)

        report = classification_report(test_labels_encoded, y_pred, output_dict=True)
        df_r = pd.DataFrame(report)
        df_r = df_r.transpose()
        #df_r.to_csv(f'/home/users/ubaldi/TESI_PA/result_CV/report_{name}/report_{i}')

        outname = f'report_{i}.csv'

        outdir = f'/home/users/ubaldi/TESI_PA/result_score/Public/{folder}/report_{name}_{str(abbr_scaler)}_YES_NO'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        fullname_r = os.path.join(outdir, outname)    

        df_r.to_csv(fullname_r)


    #mean value and std

    mean_train_score = np.mean(tot_train_score)
    mean_test_score = np.mean(tot_test_score)
    mean_weighted_ovr = np.mean(tot_weighted_ovr)

    std_train_score = np.std(tot_train_score)
    std_test_score = np.std(tot_test_score)
    std_weighted_ovr = np.std(tot_weighted_ovr)


    # pandas can convert a list of lists to a dataframe.
    # each list is a row thus after constructing the dataframe
    # transpose is applied to get to the user's desired output. 
    df = pd.DataFrame([tot_train_score, [mean_train_score], [std_train_score], 
                    tot_test_score, [mean_test_score], [std_test_score], 
                    tot_weighted_ovr, [mean_weighted_ovr], [std_weighted_ovr],
                    [scaler_], [None], ['default'], ['default'], ['default']])
    df = df.transpose() 

    fieldnames = ['train_accuracy', 'train_accuracy_MEAN', 'train_accuracy_STD',
                'test_accuracy', 'test_accuracy_MEAN', 'test_accuracy_STD',
                'roc_auc_score_weighted_ovr', 'roc_auc_score_weighted_ovr_MEAN', 'roc_auc_score_weighted_ovr_STD',
                'SCALER', 'PCA__n_components', 'CLF__algorithm', 'CLF__lr', 'CLF__n_estimators']

    ## write the data to the specified output path: "output"/+file_name
    ## without adding the index of the dataframe to the output 
    ## and without adding a header to the output. 
    ## => these parameters are added to be fit the desired output. 
    #df.to_csv(f'/home/users/ubaldi/TESI_PA/result_score/Public/score_{name}.csv', index=False, header=fieldnames)


    #create folder and save

    import os

    outname = f'score_{name}_{str(abbr_scaler)}_YES_NO.csv'

    outdir = f'/home/users/ubaldi/TESI_PA/result_score/Public/{folder}/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    

    df.to_csv(fullname, index=False, header=fieldnames)



create_csv_score_YES_NO(MinMaxScaler(), 'MMS')
create_csv_score_YES_NO(StandardScaler(), 'STDS')
create_csv_score_YES_NO(RobustScaler(), 'RBT')