#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:10:07 2020

@author: leonardo
"""

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


value = [ 0.73, 0.54,  
          0.79, 0.52, 
          1.00, 0.55, 
          0.69, 0.58, 
          0.79, 0.52, 
          0.62, 0.64,
         ]

std = [ 0.05, 0.08,  
        0.07, 0.07, 
        0.00, 0.10, 
        0.05, 0.11, 
        0.13, 0.06, 
        0.06, 0.10, 
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


outname = f'3H_AUC_PUBLIC_TRAIN_TEST.pdf'

outdir = '/home/leonardo/Scrivania/Presentazione/img/original_create_da_me_python/PA/3H/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)    



ax.figure.set_size_inches(7,5)
ax.figure.tight_layout()

ax.figure.savefig(fullname)

