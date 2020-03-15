import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import os
import ANOVA_features_selection


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

#public_labels = df_train.Histology
#PA_labels = df_test.Histology

#tot_data = pd.concat([public_data, PA_data], axis=0)
#tot_label = pd.concat([public_labels, PA_labels], axis=0)

n_bins = 15

#tot_A = tot_data[tot_data['Histology'] == 'adenocarcinoma']
#tot_L = tot_data[tot_data['Histology'] == 'large cell']
#tot_S = tot_data[tot_data['Histology'] == 'squamous cell carcinoma']

pu_A = public_data[public_data['Histology'] == 'adenocarcinoma']
pu_L = public_data[public_data['Histology'] == 'large cell']
pu_S = public_data[public_data['Histology'] == 'squamous cell carcinoma']

features_selected_ANOVA = ANOVA_features_selection.f_select_ANOVA(public_data, 0.05)

pu_A = pu_A[features_selected_ANOVA]
pu_L = pu_L[features_selected_ANOVA]
pu_S = pu_S[features_selected_ANOVA]

#pa_A = PA_data[PA_data['Histology'] == 'adenocarcinoma']
#pa_L = PA_data[PA_data['Histology'] == 'large cell']
#pa_S = PA_data[PA_data['Histology'] == 'squamous cell carcinoma']

for column in pu_A.columns:

    #feat_tot = tot_data.iloc[:, i]
    #feat_tot_A = tot_A.iloc[:, i]
    #feat_tot_L = tot_L.iloc[:, i]
    #feat_tot_S = tot_S.iloc[:, i]

    feat_pu = public_data[column]
    feat_pu_A = pu_A[column]
    feat_pu_L = pu_L[column]
    feat_pu_S = pu_S[column]

    #feat_pa_A = pa_A.iloc[:, i]
    #feat_pa_L = pa_L.iloc[:, i]
    #feat_pa_S = pa_S.iloc[:, i]

    MAX = max(feat_pu)
    MIN = min(feat_pu)

    bin_edges = np.linspace(MIN, MAX, n_bins)

    fig, (ax1) = plt.subplots(1, 1)

    ax1.hist(feat_pu_A, bins=bin_edges, histtype='step', alpha=0.8, linewidth=3, label='adenocarcinoma')
    ax1.hist(feat_pu_L, bins=bin_edges, histtype='step', alpha=0.8, linewidth=3, label='large cell')
    ax1.hist(feat_pu_S, bins=bin_edges, histtype='step', alpha=0.8, linewidth=3, label='squamous cell carcinoma')
    ax1.set_title('Public dataset')


    #ax2.hist(feat_pa_A, bins=bin_edges, histtype='step', alpha=0.8, linewidth=3)
    #ax2.hist(feat_pa_L, bins=bin_edges, histtype='step', alpha=0.8, linewidth=3)
    #ax2.hist(feat_pa_S, bins=bin_edges, histtype='step', alpha=0.8, linewidth=3)
    #ax2.set_title('PA dataset')


    #ax3.hist(feat_tot_A, bins=bin_edges, histtype='step', alpha=0.8, linewidth=3)
    #ax3.hist(feat_tot_L, bins=bin_edges, histtype='step', alpha=0.8, linewidth=3)
    #ax3.hist(feat_tot_S, bins=bin_edges, histtype='step', alpha=0.8, linewidth=3)
    #ax3.set_title('Merged dataset')


    fig.suptitle(f'{column}', horizontalalignment='center', fontsize=16)
    fig.legend(prop={'size':6}, loc='lower center')
    fig.set_figwidth(6)
    fig.set_figheight(8)


    #create folder and save

    outname = f'histogram_distribution_anova_features_{column}.png'

    outdir = '/home/users/ubaldi/TESI_PA/plot/Public/ANOVA_features'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    

    plt.savefig(fullname)
    plt.close(fig)



    #plt.savefig(f'/home/users/ubaldi/TESI_PA/plot/total_compare/compare_histogram_distribution_features_{tot_data.columns[i]}.png')
    #plt.close(fig)