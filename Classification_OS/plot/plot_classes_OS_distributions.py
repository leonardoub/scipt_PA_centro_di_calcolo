import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import os


#load data

train_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_training2.csv'
test_dataset_path = '/home/users/ubaldi/TESI_PA/data/database_nostro.csv'

df_train = pd.read_csv(train_dataset_path)
df_test = pd.read_csv(test_dataset_path)

df_train.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)
df_test.rename(columns={'Survival.time (months)':'Surv_time_months'}, inplace=True)

df_train.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)
df_test.rename(columns={'Overall.Stage':'Overall_Stage'}, inplace=True)

public_data = df_train.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event'], axis=1)
PA_data = df_test.drop(['Histology', 'Surv_time_months', 'OS', 'deadstatus.event'], axis=1)

public_labels = df_train.Overall_Stage
PA_labels = df_test.Overall_Stage

tot_data = pd.concat([public_data, PA_data], axis=0)
tot_label = pd.concat([public_labels, PA_labels], axis=0)

n_bins = 15

tot_I = tot_data[tot_data['Overall_Stage'] == 'I']
tot_II = tot_data[tot_data['Overall_Stage'] == 'II']
tot_IIIa = tot_data[tot_data['Overall_Stage'] == 'IIIa']
tot_IIIb = tot_data[tot_data['Overall_Stage'] == 'IIIb']


pu_I = public_data[public_data['Overall_Stage'] == 'I']
pu_II = public_data[public_data['Overall_Stage'] == 'II']
pu_IIIa = public_data[public_data['Overall_Stage'] == 'IIIa']
pu_IIIb = public_data[public_data['Overall_Stage'] == 'IIIb']


pa_I = PA_data[PA_data['Overall_Stage'] == 'I']
pa_II = PA_data[PA_data['Overall_Stage'] == 'II']
pa_IIIa = PA_data[PA_data['Overall_Stage'] == 'IIIa']
pa_IIIb = PA_data[PA_data['Overall_Stage'] == 'IIIb']


for column in tot_data.column:

    if column != 'Overall_Stage'

        feat_tot = tot_data[column]
        feat_tot_I = tot_I[column]
        feat_tot_II = tot_II[column]
        feat_tot_IIIa = tot_IIIa[column]
        feat_tot_IIIb = tot_IIIb[column]


        feat_pu_I = pu_I[column]
        feat_pu_II = pu_II[column]
        feat_pu_IIIa = pu_IIIa[column]
        feat_pu_IIIb = pu_IIIb[column]


        feat_pa_I = pa_I[column]
        feat_pa_II = pa_II[column]
        feat_pa_IIIa = pa_IIIa[column]
        feat_pa_IIIb = pa_IIIb[column]


        MAX = max(feat_tot)
        MIN = min(feat_tot)

        bin_edges = np.linspace(MIN, MAX, n_bins)

        hist_type = 'step'
        alpha_value = 0.7

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        ax1.hist(feat_pu_I, bins=bin_edges, histtype=hist_type, alpha=alpha_value, linewidth=3, label='I')
        ax1.hist(feat_pu_II, bins=bin_edges, histtype=hist_type, alpha=alpha_value, linewidth=3, label='II')
        ax1.hist(feat_pu_IIIa, bins=bin_edges, histtype=hist_type, alpha=alpha_value, linewidth=3, label='IIIa')
        ax1.hist(feat_pu_IIIb, bins=bin_edges, histtype=hist_type, alpha=alpha_value, linewidth=3, label='IIIb')
        ax1.set_title('Public dataset')


        ax2.hist(feat_pa_I, bins=bin_edges, histtype=hist_type, alpha=alpha_value, linewidth=3)
        ax2.hist(feat_pa_II, bins=bin_edges, histtype=hist_type, alpha=alpha_value, linewidth=3)
        ax2.hist(feat_pa_IIIa, bins=bin_edges, histtype=hist_type, alpha=alpha_value, linewidth=3)
        ax2.hist(feat_pa_IIIb, bins=bin_edges, histtype=hist_type, alpha=alpha_value, linewidth=3)
        ax2.set_title('PA dataset')



        ax3.hist(feat_tot_I, bins=bin_edges, histtype=hist_type, alpha=alpha_value, linewidth=3)
        ax3.hist(feat_tot_II, bins=bin_edges, histtype=hist_type, alpha=alpha_value, linewidth=3)
        ax3.hist(feat_tot_IIIa, bins=bin_edges, histtype=hist_type, alpha=alpha_value, linewidth=3)
        ax3.hist(feat_tot_IIIb, bins=bin_edges, histtype=hist_type, alpha=alpha_value, linewidth=3)
        ax3.set_title('Merged dataset')


        fig.suptitle(f'{column}', horizontalalignment='center', fontsize=16)
        fig.legend()
        fig.set_figwidth(15)
        fig.set_figheight(6)


        #create folder and save

        outname = f'compare_histogram_distribution_features_{column}.png'

        outdir = '/home/users/ubaldi/TESI_PA/plot/total_compare/OS_classes'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        fullname = os.path.join(outdir, outname)    

        plt.savefig(fullname)
        plt.close(fig)



    #plt.savefig(f'/home/users/ubaldi/TESI_PA/plot/total_compare/compare_histogram_distribution_features_{tot_data.columns[i]}.png')
    #plt.close(fig)