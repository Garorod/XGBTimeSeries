# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


df = pd.read_parquet('ep_split_ca_10.parquet.60_day_delinq_trim')
df['North'] = df['attom_site_zip'] > 93000
df = df.drop(['attom_site_zip','sr_lender_type_1'
                  ,'original_house_price_index'],axis = 1)
df = df[df.sr_date_transfer> 20090101]
#df = df[df['payment_current_status'].isin(['C','P',"1","2"])]

#%%
df['target'] = np.where(df.payment_current_status == '2',1,0)
df = df[['dum_gse','dum_ginnie','mcd_creditscore','mcd_orig_ltv','mcd_orig_rate',
        'mcd_orig_term','mcd_originalpropertyvalue', 'dum_bank','dum_nonbank',
        'month_count','current_loan_balance','house_price_index',
        'coupon_differential','current_property_value','delinq_propensity',
        'North','target']]

df = df.dropna()

X = df[['dum_gse','dum_ginnie','mcd_creditscore','mcd_orig_ltv','mcd_orig_rate',
        'mcd_orig_term','mcd_originalpropertyvalue', 'dum_bank','dum_nonbank',
        'month_count','current_loan_balance',
        'coupon_differential','current_property_value',
        'North']]
y = df[["target"]]

rf = RandomForestClassifier(class_weight = 'balanced',
                            max_depth=5,n_estimators = 10,
                            warm_start = True,n_jobs = -1)
rf.fit(X,y)


probs = rf.predict_proba(X)
preds = probs[:,1]
y_test = y
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


features = X.columns
importances = rf.feature_importances_
indices = np.argsort(importances)
indices = indices[:10]
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


y_score = preds
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(classifier, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))


















