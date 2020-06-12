import os
import pandas as pd

def function_save_output(final_data, name_clf):
    outname = f'best_params_{name_clf}_PU_PA_regression_ST.csv'
    outdir = f'/home/users/ubaldi/TESI_PA/result_CV_bis/ST_regression/train_PU_test_PA/GSCV/MAE/{name_clf}_test_stability'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    

    final_data.to_csv(fullname)
