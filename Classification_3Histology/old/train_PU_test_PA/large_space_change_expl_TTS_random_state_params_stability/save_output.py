import os
import pandas as pd

def function_save_output(final_data, name_clf):
    outname = f'best_params_{name_clf}_3_classes_TRAIN_PU_TEST_PA.csv'
    outdir = f'/home/users/ubaldi/TESI_PA/result_CV/3_classes_H/Train_PU_Test_PA/large_space_change_expl_TTS_rand_state/{name_clf}_test_stability'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    

    final_data.to_csv(fullname)