import pandas as pd
from sklearn.preprocessing import LabelEncoder


def function_load_data_2_class():

    #load data

    train_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_training2_without_nan_OS.csv'
    test_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_nostro.csv'

    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)

    df_train.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)
    df_test.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)

    df_train.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)
    df_test.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)


    #considerare i casi del database di Public che hanno Overall_Stage = I o II
    df_train_selected = df_train.loc[(df_train['Overall_Stage']=='I') | (df_train['Overall_Stage']=='II')]


    public_data = df_train_selected.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)
    PA_data = df_test.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)

    public_labels = df_train_selected.Overall_Stage
    PA_labels = df_test.Overall_Stage


    #Vettorizzare i label

    encoder = LabelEncoder()


    public_labels_encoded = encoder.fit_transform(public_labels)
    PA_labels_encoded = encoder.transform(PA_labels)


    return public_data, public_labels_encoded, PA_data, PA_labels_encoded
