import os
import pandas as pd

def function_save_output(final_data, name_clf):
    outname = f'best_params_{name_clf}_PU_regression_ST.csv'
    outdir = f'/home/leonardo/Scrivania/result_CV_bis/ST_regression/PUBLIC/nested_cv/MAE/{name_clf}_test_stability'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    

    final_data.to_csv(fullname)
