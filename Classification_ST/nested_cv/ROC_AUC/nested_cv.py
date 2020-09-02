from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd



def function_nested_cv(data, labels, pipel, grid_params, rs_outer_kf):

    df = pd.DataFrame()
    best_est_dict = {}

    i = 0
    
    #Vettorizzare i label
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs_outer_kf)


    # Looping through the outer loop, feeding each training set into a GSCV as the inner loop
    for train_index, test_index in outer_kf.split(data, labels_encoded):
        
        i+=1

        GSCV = GridSearchCV(pipel, param_grid=grid_params, cv=inner_kf, n_jobs=-1, scoring=['roc_auc', 'accuracy'], refit='roc_auc', verbose=1)
        
        # GSCV is looping through the training data to find the best parameters. This is the inner loop
        GSCV.fit(data.iloc[train_index, :], labels_encoded[train_index])
        
        # The best hyper parameters from GSCV is now being tested on the unseen outer loop test data.
        pred_test = GSCV.predict(data.iloc[test_index, :])
        
        #per prova
        pred_train = GSCV.predict(data.iloc[train_index, :])

        pred_proba_train = GSCV.predict_proba(data.iloc[train_index, :])[:, 1]
        
        pred_proba_test = GSCV.predict_proba(data.iloc[test_index, :])[:, 1]
        

        #per far uscire i best_estimators in qualche modo
        best_est_dict.update({f'best_est_{i}' : GSCV.best_estimator_})

       
        # Appending the "winning" hyper parameters and their associated accuracy score
        
        best_p = GSCV.best_params_

        bp = pd.DataFrame(best_p, index=[i])
        bp['inner_loop_roc_auc_scores'] = GSCV.best_score_
        
        #ROC AUC WITHOUT predict proba: WRONG WAY
        bp['train_roc_auc_scores'] = roc_auc_score(labels_encoded[train_index], pred_train)
        bp['outer_loop_roc_auc_scores'] = roc_auc_score(labels_encoded[test_index], pred_test)
        
        #ROC AUC WITH PREDICT PROBA
        bp['train_roc_auc_scores_predict_proba'] = roc_auc_score(labels_encoded[train_index], pred_proba_train)
        bp['outer_loop_roc_auc_scores_predict_proba'] = roc_auc_score(labels_encoded[test_index], pred_proba_test)
       
        bp['train_accuracy_scores'] = accuracy_score(labels_encoded[train_index], pred_train)
        bp['outer_loop_accuracy_scores'] = accuracy_score(labels_encoded[test_index], pred_test)

        bp['train_balanced_accuracy_scores'] = balanced_accuracy_score(labels_encoded[train_index], pred_train)
        bp['outer_loop_balanced_accuracy_scores'] = balanced_accuracy_score(labels_encoded[test_index], pred_test)

        bp['random_state_clf'] = 503
        bp['random_state_PCA'] = 42
        bp['random_state_inner_kf'] = 1
        bp['random_state_outer_kf'] = rs_outer_kf


        df = df.append(bp, ignore_index=True)



    return df, best_est_dict
