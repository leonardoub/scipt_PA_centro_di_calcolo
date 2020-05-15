import os
import pandas as pd

def function_save_output(final_data, name_clf):
    outname = f'best_params_{name_clf}_OS_1_2_TRAIN_PA_TEST_PU.csv'
    outdir = f'/home/users/ubaldi/TESI_PA/result_CV/2_classes_OS//Train_PA_Test_PU_OS_1_2/{name_clf}_test_stability'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    

    final_data.to_csv(fullname)