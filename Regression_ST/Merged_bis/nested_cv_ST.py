from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd



def function_nested_cv_ST(data, labels, pipel, grid_params):

    df = pd.DataFrame()


    i = 0


    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    inner_kf = KFold(n_splits=5, shuffle=True, random_state=1)
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=2)


    # Looping through the outer loop, feeding each training set into a GSCV as the inner loop
    for train_index, test_index in outer_kf.split(data, labels):
        
        i+=1

        GSCV = GridSearchCV(pipel, param_grid=grid_params, cv=inner_kf, n_jobs=-1, scoring='neg_mean_absolute_error', 
               refit='neg_mean_absolute_error', verbose=1)
        
        # GSCV is looping through the training data to find the best parameters. This is the inner loop
        GSCV.fit(data.iloc[train_index, :], labels[train_index])
        
       
        # Appending the "winning" hyper parameters and their associated accuracy score
        
        best_p = GSCV.best_params_

        score_train = GSCV.score(data.iloc[train_index, :], labels[train_index])
        score_test = GSCV.score(data.iloc[test_index, :], labels[test_index])


        bp = pd.DataFrame(best_p, index=[i])
               
        #adding scores
        bp['MAE_train'] = -score_train
        bp['MAE_test'] = -score_test
        

        bp['random_state_clf'] = 503
        bp['random_state_PCA'] = 42
        bp['random_state_inner_kf'] = 1
        bp['random_state_outer_kf'] = 2

        df = df.append(bp, ignore_index=True)



    return df
