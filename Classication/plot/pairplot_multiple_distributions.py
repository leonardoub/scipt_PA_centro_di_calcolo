import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import scipy
import seaborn as sns



#load data

train_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_training2.csv'
test_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_nostro_without_nan.csv'

df_train = pd.read_csv(train_dataset_path)
df_test = pd.read_csv(test_dataset_path)

df_train.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)
df_test.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)


df_train.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)
df_test.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)

public_data = df_train.drop(['OS', 'deadstatus.event','Overall_Stage'], axis=1)
PA_data = df_test.drop(['OS', 'deadstatus.event','Overall_Stage'], axis=1)

public_labels = df_train.Histology
PA_labels = df_test.Histology


sns_plot = sns.pairplot(public_data, hue='Histology')
sns_plot.savefig("/home/users/ubaldi/TESI_PA/plot/pairplot_multiple_distribution.png")

