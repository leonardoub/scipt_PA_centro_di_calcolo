import os
import pandas as pd

def function_save_output(final_data, scaler, name_clf, rs_outer_kf):
    outname = f'score_default_HP_RFoKF_{rs_outer_kf}_{name_clf}_{scaler}.csv'

    outdir = f'/home/users/ubaldi/TESI_BRATS/result_score/data_without_HISTO_SPATIAL_INTENSITY/score_default_HP/RS_outer_KF_{rs_outer_kf}/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    
    final_data.to_csv(fullname)





