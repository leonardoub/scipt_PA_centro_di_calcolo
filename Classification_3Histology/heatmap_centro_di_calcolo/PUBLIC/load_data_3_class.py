import pandas as pd
from sklearn.preprocessing import LabelEncoder


def function_load_data_3_class():


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
    #PA_data = df_test.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)

    public_labels = df_train.Histology
    #PA_labels = df_test.Histology

    #tot_data = pd.concat([public_data, PA_data], axis=0)
    #tot_label = pd.concat([public_labels, PA_labels], axis=0)


    return public_data, public_labels