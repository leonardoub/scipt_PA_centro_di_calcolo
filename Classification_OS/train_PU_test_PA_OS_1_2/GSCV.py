from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd



def function_GSCV(X_train, y_train, X_test, y_test, pipel, grid_params):


    for i in range(1, 11):
        
        inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i*42)

        GSCV = GridSearchCV(pipel, param_grid=grid_params, cv=inner_kf, n_jobs=-1, scoring=['roc_auc', 'accuracy'], refit='roc_auc', verbose=1)
        
        GSCV.fit(X_train, y_train)
               
        pred_train = GSCV.predict(X_train)
        pred_test = GSCV.predict(X_test)
                
        
        pred_proba_train = GSCV.predict_proba(X_train)[:, 1]
        pred_proba_test = GSCV.predict_proba(X_test)[:, 1]
      
        # Appending the "winning" hyper parameters and their associated accuracy score
        
        best_p = GSCV.best_params_
        bp = pd.DataFrame(best_p, index=[i])

        
        
        #ROC AUC WITH PREDICT PROBA
        bp['train_roc_auc_scores_predict_proba'] = roc_auc_score(y_train, pred_proba_train)
        bp['test_loop_roc_auc_scores_predict_proba'] = roc_auc_score(y_test, pred_proba_test)
       
        bp['train_accuracy_scores'] = accuracy_score(y_train, pred_train)
        bp['outer_loop_accuracy_scores'] = accuracy_score(y_test, pred_test)

        bp['train_balanced_accuracy_scores'] = balanced_accuracy_score(y_train, pred_train)
        bp['outer_loop_balanced_accuracy_scores'] = balanced_accuracy_score(y_test, pred_test)

        bp['random_state_kf'] = i*42

        df = df.append(bp, ignore_index=True)



    return df
