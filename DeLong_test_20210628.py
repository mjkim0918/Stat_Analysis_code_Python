import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as st
from sklearn import metrics
import math


#DeLong's test
#actual : Ground Truth, preds_A : AI score of model A, preds_B : AI score of model B
def delong_test(actual, preds_A, preds_B):
    def kernel(X, Y):
        return .5 if Y==X else int(Y < X)
    def auc(X, Y):
        return 1/(len(X)*len(Y)) * sum([kernel(x, y) for x in X for y in Y])

    def structural_components(X, Y):
        V10 = [1/len(Y) * sum([kernel(x, y) for y in Y]) for x in X]
        V01 = [1/len(X) * sum([kernel(x, y) for x in X]) for y in Y]
        return V10, V01
            
    def get_S_entry(V_A, V_B, auc_A, auc_B):
        return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])
    def z_score(var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB)**(.5))


    # Model A  vs. model B
    def group_preds_by_label(preds, actual):
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y
        
    X_A, Y_A = group_preds_by_label(preds_A, actual)
    X_B, Y_B = group_preds_by_label(preds_B, actual)
    V_A10, V_A01 = structural_components(X_A, Y_A)
    V_B10, V_B01 = structural_components(X_B, Y_B)
    auc_A = auc(X_A, Y_A)
    auc_B = auc(X_B, Y_B)
    
    # Compute entries of covariance matrix S (covar_AB = covar_BA)
    var_A = (get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)
            + get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
    var_B = (get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)
            + get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
    covar_AB = (get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)
                + get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))
    
    # Two tailed test
    z = z_score(var_A, var_B, covar_AB, auc_A, auc_B)
    p = st.norm.sf(abs(z))*2

    # 95% confidence interval (DeLong Method)
    LL_A = auc_A - st.norm.ppf(0.975)*math.sqrt(var_A)
    UL_A = auc_A + st.norm.ppf(0.975)*math.sqrt(var_A)
    LL_B = auc_B - st.norm.ppf(0.975)*math.sqrt(var_B)
    UL_B = auc_B + st.norm.ppf(0.975)*math.sqrt(var_B)

    return  print("AUC of model A : {0}({1}, {2}), \nAUC of model B : {3}({4}, {5}),\nP value : {6}".format(round(auc_A,4), round(LL_A,4), round(UL_A,4), round(auc_B,4), round(LL_B,4), round(UL_B,4), round(p,4)))


data = pd.read_csv('/Users/lunit/Documents/Statistical Analysis/RSNA/RSNA 2021/SRM/data/vega_dbt.csv')
actual = (data['Label'] == "cancer" ) * 1 
preds_A = data['BaselineScore']
preds_B = data['SRMScore']

delong_test(actual, preds_A, preds_B)


