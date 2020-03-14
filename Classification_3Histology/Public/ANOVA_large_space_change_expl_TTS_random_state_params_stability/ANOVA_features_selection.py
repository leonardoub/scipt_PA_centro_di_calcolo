# load packages
import scipy.stats as stats



#Function for features selection based on ANOVA

def f_select_ANOVA(data_with_hist, alpha):

    data_1 = data_with_hist.drop(['Histology'], axis=1)

    features_selected_ANOVA = []

    for column in data_1.columns:

        feat_A = data_with_hist[column][data_with_hist.Histology == 'adenocarcinoma']
        feat_L = data_with_hist[column][data_with_hist.Histology == 'large cell']
        feat_S = data_with_hist[column][data_with_hist.Histology == 'squamous cell carcinoma']

        fvalue, pvalue = stats.f_oneway(feat_A, feat_L, feat_S)

        if pvalue <= alpha:
            features_selected_ANOVA.append(column)

    return features_selected_ANOVA