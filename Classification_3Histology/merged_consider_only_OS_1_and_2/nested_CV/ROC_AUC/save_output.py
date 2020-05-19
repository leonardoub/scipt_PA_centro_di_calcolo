import os
import pandas as pd

def function_save_output(final_data, name_clf):
    outname = f'best_params_{name_clf}_3_classes.csv'
    outdir = f'/home/users/ubaldi/TESI_PA/result_CV/3_classes_H/merged_consider_only_OS_1_and_2/nested_cv/ROC_auc/{name_clf}_test_stability'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullname = os.path.join(outdir, outname)    

    final_data.to_csv(fullname)