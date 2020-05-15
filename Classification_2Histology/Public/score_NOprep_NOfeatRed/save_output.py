import os
import pandas as pd

def function_save_output(final_data, name_clf):
    outname = f'score_{name_clf}_NO_NO.csv'
    outdir = f'/home/users/ubaldi/TESI_PA/result_score/Public/score_NOprep_NOfeatRed_2H'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    

    final_data.to_csv(fullname)