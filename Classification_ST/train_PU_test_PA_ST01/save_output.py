import os
import pandas as pd

def function_save_output(final_data, name_clf):
    outname = f'best_params_{name_clf}_ST_TRAIN_PU_TEST_PA.csv'
    outdir = f'/home/users/ubaldi/TESI_PA/result_CV_bis/2_classes_ST/Train_PU_Test_PA_ST/{name_clf}_test_stability'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    

    final_data.to_csv(fullname)
