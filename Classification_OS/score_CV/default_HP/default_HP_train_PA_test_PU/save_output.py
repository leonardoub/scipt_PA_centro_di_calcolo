import os
import pandas as pd

def function_save_output(final_data, scaler, name_clf):
    outname = f'score_default_HP_2OS_PA_PU_{name_clf}_{scaler}.csv'

    outdir = f'/home/users/ubaldi/TESI_PA/score_ROC_AUC_default_HP/2OS/train_PA_test_PU/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    
    final_data.to_csv(fullname)





