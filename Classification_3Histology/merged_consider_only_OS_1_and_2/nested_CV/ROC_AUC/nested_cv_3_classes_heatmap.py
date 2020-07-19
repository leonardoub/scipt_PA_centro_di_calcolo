from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

def function_nested_cv_3_classes(data, labels, pipel, grid_params):

    df = pd.DataFrame()


    i = 0
    
    #Vettorizzare i label
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)


    # Looping through the outer loop, feeding each training set into a GSCV as the inner loop
    for train_index, test_index in outer_kf.split(data, labels_encoded):
        
        i+=1

        GSCV = GridSearchCV(pipel, param_grid=grid_params, cv=inner_kf, n_jobs=-1, scoring='roc_auc_ovr_weighted', 
               refit='roc_auc_ovr_weighted', verbose=1, return_train_score=True)
        
        # GSCV is looping through the training data to find the best parameters. This is the inner loop
        GSCV.fit(data.iloc[train_index, :], labels_encoded[train_index])
        
        df_gridsearch = pd.DataFrame(GSCV.cv_results_)
    
    
        max_scores = df_gridsearch.groupby(['param_clf__n_estimators', 
                                            'param_clf__max_depth']).max()
        
        #TRAIN
        max_scores = max_scores.unstack()[['mean_test_score', 'mean_train_score']]
        sns_plot = sns.heatmap(max_scores.mean_train_score, annot=True, fmt='.2g')
        
        
        
        
        
        #create folder and save
        
        
        outname = f'heatmap_RF_3H_MERGED_OS12_TRAIN_{i}.png'
        outdir = f'/home/users/ubaldi/TESI_PA/fig_heatmap_tris/PA/3H/MERGED_OS12/RF_LARGE_SPACE'
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        fullname = os.path.join(outdir, outname)    
        
        
        sns_plot.figure.set_size_inches(7,5)
        sns_plot.figure.tight_layout()
        
        sns_plot.figure.savefig(fullname)
        plt.close()
        
        #TEST
        
        sns_plot = sns.heatmap(max_scores.mean_test_score, annot=True, fmt='.2g')
        
        
        #create folder and save
        
        
        
        outname = f'heatmap_RF_3H_MERGED_OS12_TEST_{i}.png'
        outdir = f'/home/users/ubaldi/TESI_PA/fig_heatmap_tris/PA/3H/MERGED_OS12/RF_LARGE_SPACE'
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        fullname = os.path.join(outdir, outname)    
        
        sns_plot.figure.set_size_inches(7,5)
        sns_plot.figure.tight_layout()
        
        sns_plot.figure.savefig(fullname)
        plt.close()


        # The best hyper parameters from GSCV is now being tested on the unseen outer loop test data.
        pred_test = GSCV.predict(data.iloc[test_index, :])
        
        #per prova
        pred_train = GSCV.predict(data.iloc[train_index, :])

        pred_proba_train = GSCV.predict_proba(data.iloc[train_index, :])
        
        pred_proba_test = GSCV.predict_proba(data.iloc[test_index, :])


        #per far uscire i best_estimators in qualche modo
        #best_est_dict.update({f'best_est_{i}' : GSCV.best_estimator_})

      
        # Appending the "winning" hyper parameters and their associated accuracy score
        
        best_p = GSCV.best_params_

        bp = pd.DataFrame(best_p, index=[i])
        bp['inner_loop_roc_auc_ovr_weigthed_scores'] = GSCV.best_score_
        
        
        #ROC AUC OVR WEIGHTED WITH PREDICT PROBA
        bp['train_roc_auc_ovr_weigthed_scores_predict_proba'] = roc_auc_score(labels_encoded[train_index], pred_proba_train, average='weighted',  multi_class='ovr')
        bp['outer_loop_roc_auc_ovr_weigthed_scores_predict_proba'] = roc_auc_score(labels_encoded[test_index], pred_proba_test, average='weighted',  multi_class='ovr')
       
        bp['train_balanced_accuracy_scores'] = balanced_accuracy_score(labels_encoded[train_index], pred_train)
        bp['outer_loop_balanced_accuracy_scores'] = balanced_accuracy_score(labels_encoded[test_index], pred_test)

        bp['random_state_clf'] = 503
        bp['random_state_PCA'] = 42
        bp['random_state_inner_kf'] = 1
        bp['random_state_outer_kf'] = 2

        df = df.append(bp, ignore_index=True)



    return df
