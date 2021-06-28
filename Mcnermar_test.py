import numpy as np
import pandas as pd

# Example of calculating the mcnemar test
from statsmodels.stats.contingency_tables import mcnemar

#Mcnemar's test
#data : data set
#actual : Ground Truth (should be coded as 0,1)
#label_A : predicted label of model A (if preds_A > threshold then label_A ==1 else label_A == 0)  (should be coded as 0,1)
#label_B : predicted label of model B (if preds_B > threshold then label_B ==1 else label_B == 0)  (should be coded as 0,1)

def mcnemar_sesp(data, actual, label_A, label_B):

    data_sen = data[data[actual] == 1]
    data_spec = data[data[actual] == 0]

    sen_crosstab = pd.crosstab(data_sen[label_A], data_sen[label_B],margins = False)
    spec_crosstab = pd.crosstab(data_spec[label_A], data_spec[label_B],margins = False)


    # calculate mcnemar test
    sen_result = mcnemar(sen_crosstab, exact=False)
    spec_result = mcnemar(spec_crosstab, exact=False)

    # summarize the finding
    return print("Mcnemar's test for sensitivity : statistic=%.3f, p-value=%.3f" % (sen_result.statistic, sen_result.pvalue), "\n"
                 "Mcnemar's test for specificity : statistic=%.3f, p-value=%.3f" % (spec_result.statistic, spec_result.pvalue))




data = pd.read_csv('/Users/lunit/Documents/Statistical Analysis/RSNA/RSNA 2021/SRM/data/vega_dbt.csv')
data.head()
data['base_label']=''
data['SRMS_label']=''
data['Label2']=''

for i in range(0,len(data)):
    data['base_label'][i] = 1 if data['BaselineScore'][i] > 0.3 else 0
    data['SRMS_label'][i] = 1 if data['SRMScore'][i] > 0.3 else 0
    data['Label2'][i] = 1 if data['Label'][i] == 'cancer' else 0

data.head()


mcnemar_sesp(data,'Label2','base_label','SRMS_label')

