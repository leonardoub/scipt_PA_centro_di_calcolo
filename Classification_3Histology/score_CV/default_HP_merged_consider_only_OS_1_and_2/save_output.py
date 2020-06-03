import os
import pandas as pd

def function_save_output(final_data, scaler, name_clf, rs_outer_kf):
    outname = f'score_default_HP_3H_merged_consider_only_OS12_{name_clf}_{scaler}.csv'

    outdir = f'/home/users/ubaldi/TESI_PA/score_ROC_AUC_default_HP/3H/merged_consider_only_OS12/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    
    final_data.to_csv(fullname)





