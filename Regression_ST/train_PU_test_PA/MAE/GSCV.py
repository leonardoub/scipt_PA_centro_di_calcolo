from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd



def function_GSCV(X_train, y_train, X_test, y_test, pipel, grid_params):

    df = pd.DataFrame()


    for i in range(1, 11):
        
        inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i*42)

        
        GSCV = GridSearchCV(pipel, param_grid=grid_params, cv=inner_kf, n_jobs=-1, scoring='neg_mean_absolute_error', 
               refit='neg_mean_absolute_error', verbose=1)
        
        # GSCV is looping through the training data to find the best parameters. This is the inner loop
        GSCV.fit(X_train, y_train)        
       
        # Appending the "winning" hyper parameters and their associated accuracy score
        
        best_p = GSCV.best_params_

        score_train = GSCV.score(X_train, y_train)
        score_test = GSCV.score(X_test, y_test)


        bp = pd.DataFrame(best_p, index=[i])
               
        #adding scores
        bp['MAE_train'] = -score_train
        bp['MAE_test'] = -score_test
        

        bp['random_state_clf'] = 503
        bp['random_state_PCA'] = 42
        bp['random_state_kf'] = i*42

        df = df.append(bp, ignore_index=True)



    return df
