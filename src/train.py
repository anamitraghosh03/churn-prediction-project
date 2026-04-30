
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#data preparation 

df = pd.read_csv('data/data.csv')
df.head()


df.head().T

df.columns

df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns


categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df['customerid']

df.dtypes

df.totalcharges

df.head().T

tc = pd.to_numeric(df.totalcharges, errors='coerce')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')

df.totalcharges = df.totalcharges.fillna(0)

df[tc.isnull()][['customerid','totalcharges']]

tc

# %%
df.churn

# %%
df.churn = (df.churn == 'yes').astype(int)

# %%
df.churn.head()

# %%
from sklearn.model_selection import train_test_split

# %%
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)
len(df_full_train), len(df_test)

# %%
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)
len(df_train), len(df_val), len(df_test)

# %%
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)

# %%
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

# %%
del df_train['churn']
del df_val['churn']
del df_test['churn']

# %%
df_full_train = df_full_train.reset_index(drop = True)

# %%
df_full_train.isnull().sum()

# %%
df_full_train.churn.value_counts(normalize = True)

# %%
global_churn_rate = df_full_train.churn.mean()

# %%
round(global_churn_rate, 2)

# %%
df_full_train.dtypes

# %%
numerical = ['tenure',  'monthlycharges', 'totalcharges']


# %%
df_full_train.columns

# %%
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']

# %%
df_full_train[categorical].nunique()

# %%
df_full_train.head()

# %%
churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()
churn_male

# %%
churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
churn_female

# %%
df_full_train.head()

# %%
churn_partner_yes = df_full_train[df_full_train.partner == 'yes'].churn.mean()
churn_partner_yes

# %%
churn_partner_no = df_full_train[df_full_train.partner == 'no'].churn.mean()
churn_partner_no

# %%
global_churn = df_full_train.churn.mean()
global_churn

# %%
global_churn - churn_partner_yes

# %%
global_churn - churn_partner_no

# %%
churn_partner_no/global_churn

# %%
churn_partner_yes/global_churn

# %%
df_full_train.groupby('gender').churn.mean()

# %%
df_group = df_full_train.groupby('gender').churn.agg(['mean', 'count'])

# %%
df_group['diff'] = global_churn - df_group['mean']
df_group['risk'] = df_group['mean']/ global_churn
df_group

# %%
from IPython.display import display

# %%
for c in categorical:
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = global_churn - df_group['mean']
    df_group['risk'] = df_group['mean']/ global_churn
    display(df_group)
    print()
    print()

# %%
from sklearn.metrics import mutual_info_score

# %%
mutual_info_score(df_full_train.gender, df_full_train.churn)

# %%
mutual_info_score(df_full_train.partner, df_full_train.churn)

# %%
mutual_info_score(df_full_train.contract, df_full_train.churn)

# %%
def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)


# %%
mi = df_full_train[categorical].apply(mutual_info_churn_score)

# %%
mi.sort_values(ascending=False)

# %%
df_full_train[numerical].corrwith(df_full_train.churn)

# %%
df_full_train[df_full_train.tenure <= 2].churn.mean()

# %%
df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean()

# %%
df_full_train[df_full_train.tenure > 12].churn.mean()

# %%
from sklearn.feature_extraction import DictVectorizer


# %%
df_train[['gender', 'contract']].iloc[:100]

# %%
dicts = df_train[['gender', 'contract']].iloc[:100].to_dict(orient = 'records')

# %%
dv = DictVectorizer(sparse=False)

# %%
dv.fit(dicts)

# %%
dv.get_feature_names_out()

# %%
dv.transform(dicts)

# %%
train_dicts = df_train[categorical + numerical].to_dict(orient = 'records')

# %%
train_dicts[0]

# %%
dv = DictVectorizer(sparse=False)

# %%
dv.fit(train_dicts)

# %%
dv.get_feature_names_out()

# %%
x_train = dv.transform(train_dicts)

# %%
x_train.shape



# %%
val_dicts = df_val[categorical + numerical].to_dict(orient = 'records')

# %%
x_val = dv.transform(val_dicts)

# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace (-5, 5, 51)

# %%
z

# %%
sigmoid(z)

# %%
plt.plot(z, sigmoid(z))

# %%
def linear_regression(xi):
    result = W0
    for j in range(len(W)):
        result = result + xi[j] * W[j]

    return result

# %%
def logistic_regression(xi):
    score = W0
    for j in range(len(W)):
        score = result + xi[j] * W[j]

    result = sigmoid(score)
    return result

# %%
from sklearn.linear_model import LogisticRegression

# %%
model = LogisticRegression()
model.fit(x_train, y_train)

# %%
model.coef_[0].round(3)

# %%
model.intercept_[0]

# %%
model.predict(x_train)

# %%
y_pred = model.predict_proba(x_val)[:, 1]

# %%
y_pred

# %%
churn_decision = (y_pred >=0.5)
churn_decision

# %%
df_val[churn_decision].customerid

# %%
(y_val == churn_decision).mean()

# %%
churn_decision.astype(int)

# %%
df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val

# %%
df_pred

# %%
df_pred['correct'] = df_pred.prediction == df_pred.actual

# %%
df_pred

# %%
df_pred.correct.mean()

# %%
model.coef_.round(3)

# %%
dv.get_feature_names_out()

# %%
a = [1,2,3,4,5]
b = 'abcd'
dict(zip(a,b))

# %%
dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))

# %%
small = ['contract', 'tenure', 'monthlycharges']

# %%
df_train[small].iloc[:10].to_dict(orient = 'records')

# %%
dicts_train_small = df_train[small].to_dict(orient = 'records')

# %%
dicts_val_small = df_val[small].to_dict(orient = 'records')

# %%
dv_small = DictVectorizer(sparse=False)

# %%
x_train_small = dv_small.fit_transform(dicts_train_small)
x_val_small = dv_small.transform(dicts_val_small)

# %%
dv_small.get_feature_names_out()

# %%
model_small = LogisticRegression()

# %%
model_small.fit(x_train_small, y_train)

# %%
W0 = model_small.intercept_[0]
W0

# %%
W = model_small.coef_[0]
W.round(3)

# %%
dict(zip(dv_small.get_feature_names_out(), W.round(3)))

# %%
df_full_train

# %%
dict_full_train = df_full_train[categorical + numerical].to_dict(orient = 'records')

# %%
dict_full_train

# %%
dict_full_train[:3]

# %%
dv = DictVectorizer(sparse = False)

# %%
x_full_train = dv.fit_transform(dict_full_train)

# %%
y_full_train = df_full_train.churn.values

# %%
model = LogisticRegression()
model.fit(x_full_train, y_full_train)


# %%
dict_test = df_test[categorical + numerical].to_dict(orient = 'records')

# %%
x_test = dv.transform(dict_test)

# %%
dict_test

# %%
x_test

# %%
y_pred = model.predict_proba(x_test)[:, 1]

# %%
churn_decision_small = (y_pred >= 0.5)

# %%
(churn_decision_small == y_test).mean()

# %%
customer = dict_test[10]

# %%
x_small = dv.transform([customer])

# %%
model.predict_proba(x_small) [0,1]

# %%
y_test[10]

# %%
len(y_val)

# %%
(y_val==churn_decision).sum()

# %%
1129/1409

# %%
from sklearn.metrics import accuracy_score

# %%
y_pred = model.predict_proba(x_val)[:, 1]

# %%
thresholds = np.linspace(0,1,21)

scores = []
for t in thresholds:
    score = accuracy_score(y_val, y_pred>=t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)

# %%
scores

# %%
plt.plot(thresholds, scores)

# %%
from collections import Counter

# %%
Counter(y_pred>=1.0)

# %%
Counter(y_val)

# %%
1-y_val.mean()

# %%
actual_positive = (y_val == 1)
actual_negative = (y_val == 0)

# %%
t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)

# %%
tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()

# %%
fp, fn

# %%
confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])

confusion_matrix

# %%
(confusion_matrix / confusion_matrix.sum()).round(2)

# %%
p = tp/(tp+fp)

# %%
p

# %%
tp+fp

# %%
tp

# %%
r = tp/(tp + fn)
r

# %%
tp + fn 

# %%
tp

# %%
215/386

# %%
tpr = tp/(tp+fn)
tpr

# %%
fpr = fp/(fp + tn)
fpr

# %%
r

# %%
thresholds = np.linspace(0,1,101)
scores = []

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    scores.append((t, tp, fp, tn, fn))

# %%
columns = ['thresholds', 'tp', 'fp', 'tn', 'fn']
df_scores = pd.DataFrame(scores, columns = columns)

# %%
df_scores.head()

# %%
df_scores['tpr'] = df_scores.tp/(df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp/(df_scores.fp + df_scores.tn)

# %%
df_scores[::10]

# %%
plt.plot(df_scores.thresholds, df_scores.tpr, label = 'TPR')
plt.plot(df_scores.thresholds, df_scores.fpr, label = 'FPR')
plt.legend()

# %%
np.random.seed(1)
y_rand = np.random.uniform(0,1, size = len(y_val))

# %%
y_rand.round(3)

# %%
((y_rand >= 0.5)==y_val).mean()


def tpr_fpr_dataframe(y_val, y_pred):
    thresholds = np.linspace(0,1,101)
    scores = []

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        
        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, tn, fn))

    columns = ['thresholds', 'tp', 'fp', 'tn', 'fn']
    df_scores = pd.DataFrame(scores, columns = columns)

    df_scores['tpr'] = df_scores.tp/(df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp/(df_scores.fp + df_scores.tn)

    return df_scores

df_rand = tpr_fpr_dataframe(y_val, y_rand)



plt.plot(df_rand.thresholds, df_rand.tpr, label = 'TPR')
plt.plot(df_rand.thresholds, df_rand.fpr, label = 'FPR')
plt.legend()


num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()

num_neg, num_pos


y_ideal = np.repeat([0,1], [num_neg, num_pos])
y_ideal


y_ideal_pred = np.linspace(0,1, len(y_val))
y_ideal_pred


1-y_val.mean()


((y_ideal_pred>=0.726)== y_ideal).mean()


df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)
df_ideal


plt.plot(df_ideal.thresholds, df_ideal.tpr, label = 'TPR')
plt.plot(df_ideal.thresholds, df_ideal.fpr, label = 'FPR')
plt.legend()


plt.plot(df_scores.thresholds, df_scores.tpr, label = 'TPR')
plt.plot(df_scores.thresholds, df_scores.fpr, label = 'FPR')


#plt.plot(df_rand.thresholds, df_rand.tpr, label = 'TPR')
#plt.plot(df_rand.thresholds, df_rand.fpr, label = 'FPR')


plt.plot(df_ideal.thresholds, df_ideal.tpr, label = 'TPR', color = 'black')
plt.plot(df_ideal.thresholds, df_ideal.fpr, label = 'FPR', color = 'black')
plt.legend()


plt.figure(figsize=(5,5))

plt.plot(df_scores.fpr, df_scores.tpr, label = 'Model')
plt.plot([0,1], [0,1], label = 'Random')

plt.legend()

plt.xlabel('FPR')
plt.ylabel('TPR')

#ROC CURVE THROUGH SCIKIT LEARN LIBRARY

from sklearn.metrics import roc_curve


fpr, tpr, thresholds = roc_curve(y_val, y_pred)


plt.figure(figsize=(5,5))

plt.plot(fpr, tpr, label = 'Model')
plt.plot([0,1], [0,1], label = 'Random')

plt.legend()

plt.xlabel('FPR')
plt.ylabel('TPR')

from sklearn.metrics import auc # auc is for area under any curve

auc(fpr, tpr)

auc(df_ideal.fpr, df_ideal.tpr)

from sklearn.metrics import roc_auc_score

roc_auc_score(y_val, y_pred)

neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]


import random

n = 100000
success = 0
for i in range(n):
     pos_index = random.randint(0, len(pos)-1)
     neg_index = random.randint(0, len(neg)-1)

     if pos[pos_index] > neg[neg_index]:
         success += 1
success / n

n = 100000
np.random.seed(1)
pos_index = np.random.randint(0, len(pos), size = n)
neg_index = np.random.randint(0, len(neg), size = n)
(pos[pos_index] > neg[neg_index]).mean()

#-----------------------------------------------------------------------------------------------------------


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient = 'records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)
    return dv, model


dv, model = train(df_train, y_train, C=10)


def predict(df, dv, model):
     dicts = df[categorical + numerical].to_dict(orient = 'records')
     X = dv.transform(dicts)
     y_pred = model.predict_proba(X)[:,1]
     return y_pred

predict(df_val, dv, model)

#--------------------------------------------------------------------------------------------------

#-------------------KFOLD VALIDATION USING SCIKIT-LEARN LIBRARY------------------------------------

from sklearn.model_selection import KFold

n_splits = 5
C = 1.0
scores = []

fold = 0

kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 1)
for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train,C=C)
        y_pred = predict(df_val ,dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

        print(f'auc on fold {fold} is {auc}')
        fold = fold + 1
    
print('validation results:')
print('C=%s %.3f +- %.3f' %(C, np.mean(scores), np.std(scores)))

len(train_idx), len(val_idx)
len(df_full_train)

df_train = df_full_train.iloc[train_idx]
df_val = df_full_train.iloc[val_idx]
df_val

#-------------------------------TRAINING THE FINAL MODEL----------------------------------------------
print('training the final model:')
dv, model = train(df_full_train, df_full_train.churn.values,C=1.0)
y_pred = predict(df_test ,dv, model)

auc = roc_auc_score(y_test, y_pred)
print(f'auc = {auc}')

#-------------------------------SAVING THE MODEL----------------------------------------------

import pickle

output_file = f'model_C={C}.bin'
output_file

f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()

print(f'the model is saved to {output_file}')

#--------------------------------END OF TRAIN.PY-----------------------------------------------






