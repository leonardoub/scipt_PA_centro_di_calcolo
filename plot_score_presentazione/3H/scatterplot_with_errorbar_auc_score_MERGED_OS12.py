#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 18:14:58 2020

@author: leonardo
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = '/home/leonardo/Scrivania/result_PA/06_03/score_ROC_AUC_optimization_using_all_HP_sets_USING_MEAN/3_classes_H/summary_scores_3H_merged_consider_only_OS_1_and_2.csv'


df = pd.read_csv(path, index_col = 'Unnamed: 0')

df.drop('SVM_sigmoid_3_classes', inplace=True)


name_clf=[name[:-10] for name in df.index]
name_clf_bis=[]

for name_bis in name_clf:
    
    if name_bis.endswith('Classifier'):
        name_bis = name_bis[:-10]

    name_clf_bis.append(name_bis)


yticks=np.arange(0, 1.1, 0.1)



fig, ax = plt.subplots()


ax.scatter(df.index, df['ROC_AUC_TEST_MEAN'])


ax.set_ylim(0, 1)    

# Find the x,y coordinates for each point
x_coords = []
y_coords = []
for point_pair in ax.collections:
    for x, y in point_pair.get_offsets():
        x_coords.append(x)
        y_coords.append(y)

x_coords_sort, y_coords_sort = zip(*sorted(zip(x_coords,y_coords), reverse=False))


ax.errorbar(x_coords_sort, y_coords_sort, yerr=df['ROC_AUC_TEST_STD'], fmt=' ', 
            zorder=-1, capsize=5, capthick=2)

ax.axhline(y=0.5, color='r', linestyle='dashed')


ax.set_yticks(yticks)

ax.set_xticklabels(name_clf_bis, rotation=65, size='x-large')

ax.set_xlabel('Classifiers', size='large')
ax.set_ylabel('ROC AUC score', size='large')

#create folder and save


outname = f'3H_AUC_MERGED_OS12.png'

outdir = '/home/leonardo/Scrivania/Presentazione/img/original_create_da_me_python/PA/3H/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    



ax.figure.set_size_inches(7,6)
ax.figure.tight_layout()

ax.figure.savefig(fullname)

