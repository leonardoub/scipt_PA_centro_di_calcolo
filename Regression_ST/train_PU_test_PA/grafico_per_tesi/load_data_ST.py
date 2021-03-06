import pandas as pd
from sklearn.preprocessing import LabelEncoder


def function_load_data_ST():


    train_dataset_path = '/home/leonardo/Scrivania/TESI_PA/data/database_training2.csv'
    test_dataset_path = '/home/leonardo/Scrivania/TESI_PA/data/database_nostro.csv'

    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)

    df_train.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)
    df_test.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)

    df_train.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)
    df_test.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)

    public_data = df_train.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)
    PA_data = df_test.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event','Overall_Stage'], axis=1)

    public_labels = df_train.Surv_time_months
    PA_labels = df_test.Surv_time_months

    public_labels = public_labels.astype('float')
    PA_labels = PA_labels.astype('float')


    return public_data, public_labels, PA_data, PA_labels
