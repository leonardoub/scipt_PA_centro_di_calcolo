import pandas as pd
from sklearn.preprocessing import LabelEncoder


def function_load_data_2_class():

    #load data

    train_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_training2_ST01.csv'
    test_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_nostro_ST01.csv'

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


    #Vettorizzare i label

    encoder = LabelEncoder()


    public_labels_encoded = encoder.fit_transform(public_labels)
    PA_labels_encoded = encoder.transform(PA_labels)


    return public_data, public_labels_encoded, PA_data, PA_labels_encoded
