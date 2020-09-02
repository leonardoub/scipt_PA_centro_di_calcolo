import os
import pandas as pd

def function_save_output(final_data, name_clf, rs_outer_kf):
    outname = f'best_params_{name_clf}_RSoKF_{rs_outer_kf}.csv'

    outdir = f'/home/users/ubaldi/TESI_PA/result_CV_bis/2_classes_ST/nested_CV/ROC_AUC_optimization/{name_clf}_stability/RS_outer_KF_{rs_outer_kf}/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    

    final_data.to_csv(fullname)
