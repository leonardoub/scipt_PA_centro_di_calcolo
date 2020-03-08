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

public_data1 = public_data.drop('Histology', axis=1)

def distplot_with_hue(data=None, x=None, hue=None, row=None, col=None, legend=True, palette=None, **kwargs):
    _, bins = np.histogram(data[x].dropna())
    g = sns.FacetGrid(data, hue=hue, row=row, col=col, palette=palette)
    g.map(sns.distplot, x, **kwargs)
    if legend and (hue is not None) and (hue not in [x, row, col]):
        g.add_legend(title=hue) 
    return g


for column in public_data1.columns:
    sns_plot = distplot_with_hue(data=public_data, x=column, hue='Histology', hist=True, kde=False, 
    hist_kws={'alpha':1,'histtype':'step', 'linewidth':3}, palette={'adenocarcinoma':'r', 'large cell':'g', 'squamous cell carcinoma':'b'})

    sns_plot.savefig(f'/home/users/ubaldi/TESI_PA/plot/Public/pairplot_Public_multiple_distribution_features_{column}.png')









#sns_plot = sns.pairplot(public_data, hue='Histology')
#sns_plot.savefig("/home/users/ubaldi/TESI_PA/plot/pairplot_multiple_distribution.png")