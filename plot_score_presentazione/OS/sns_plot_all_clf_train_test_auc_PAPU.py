#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:44:03 2020

@author: leonardo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import os



clf = ['Adaboost','Adaboost',
       'KNeighbors','KNeighbors',
       'Random Forest', 'Random Forest',
       'SVM linear', 'SVM linear', 
       'SVM RBF','SVM RBF',
       'SVM sigmoid','SVM sigmoid'   
       ]


value = [ 1.00, 0.65,  
          0.93, 0.65, 
          1.00, 0.68, 
          0.58, 0.51, 
          0.90, 0.64, 
          0.73, 0.53,
         ]

std = [ 0.01, 0.06,  
        0.04, 0.05, 
        0.00, 0.06, 
        0.41, 0.15, 
        0.05, 0.04, 
        0.17, 0.15, 
       ] 

dim_red = ['TRAIN', 'TEST']*6

yticks=np.arange(0, 1.1, 0.1)


ax = sns.pointplot(x=clf, y=value, hue=dim_red, dodge=0.3, join=False, ci=None, scale=0.5)

# Find the x,y coordinates for each point
x_coords = []
y_coords = []
for point_pair in ax.collections:
    for x, y in point_pair.get_offsets():
        x_coords.append(x)
        y_coords.append(y)

x_coords_sort, y_coords_sort = zip(*sorted(zip(x_coords,y_coords), reverse=False))

# Calculate the type of error to plot as the error bars
# Make sure the order is the same as the points were looped over
##errors = tips.groupby(['smoker', 'sex']).std()['tip']
##colors = ['steelblue']*2 + ['coral']*2 + ['green']*2 + ['red']*2

colors = ['steelblue', 'coral']*6

ax.errorbar(x_coords_sort, y_coords_sort, yerr=std, ecolor=colors, fmt=' ', 
            zorder=-1)


ax.set_xticklabels(ax.get_xticklabels(), rotation=65)
ax.axhline(y=0.5, color='r', linestyle='dashed')



ax.set_xlabel('Classifiers')
ax.set_ylabel('ROC AUC score')
#ax.set_title('ROC AUC score')
ax.set_yticks(yticks)

#create folder and save


outname = f'2OS_AUC_PAPU_TRAIN_TEST.pdf'

outdir = '/home/leonardo/Scrivania/Presentazione/img/original_create_da_me_python/PA/2OS/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    



ax.figure.set_size_inches(7,5)
ax.figure.tight_layout()

ax.figure.savefig(fullname)

