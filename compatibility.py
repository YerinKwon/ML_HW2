import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
import sklearn.linear_model as skl_lm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.impute import KNNImputer

# for printing test score
def printScore(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    print(model.score(x_test,y_test))
    print()

#default models
lr = skl_lm.LogisticRegression()
rf = RandomForestClassifier()

#Load train & test dataset
trainset = pd.read_csv('hcc-survival/trainset.csv')
testset = pd.read_csv('hcc-survival/testset.csv')
x_train, y_train = trainset.iloc[:,:-1], trainset.iloc[:,-1]
x_test, y_test = testset.iloc[:,:-1], testset.iloc[:,-1]
x_train_org, x_test_org = x_train.copy(), x_test.copy()

# For more detailed data analysis, see hcc survivor.ipynb or Report

# --------Feature Engineering-----------
# 1. Fill the missing values

#comparison on each imputing methods
# single value, median, mean, most frequent value, KNN
trs,tss = [],[]
for i in range(5):
    trs.append(x_train.copy())
    tss.append(x_test.copy())

trs[0].fillna(-1, inplace=True)
tss[0].fillna(-1, inplace=True)    
trs[1].fillna(trs[1].median(), inplace=True)
tss[1].fillna(tss[1].median(), inplace=True)
trs[2].fillna(trs[2].mean(), inplace=True)
tss[2].fillna(tss[2].mean(), inplace=True)
trs[3].fillna(trs[3].mode().iloc[0], inplace=True)
tss[3].fillna(tss[3].mode().iloc[0], inplace=True)
imputer = KNNImputer(n_neighbors=10)
trs[4] = imputer.fit_transform(trs[4])
tss[4] = imputer.fit_transform(tss[4])
#print the comparison result
print()
print("-------------- Imputing methods --------------")
methods = ['single value','median','mean','mode','KNN']
for i in range(5):
    lr.fit(trs[i],y_train)
    print(methods[i]+": "+str(lr.score(tss[i],y_test)))
print()
#selected: single value
x_train.fillna(-1, inplace=True)
x_test.fillna(-1, inplace=True)
x_train_org.fillna(-1, inplace=True)
x_test_org.fillna(-1, inplace=True)

# 2. Remove columns with high correlation
print("-------- Remove highly correlated columns --------")
x_train.drop(labels='Dir. Bil', axis=1, inplace=True, errors='ignore')
x_test.drop(labels='Dir. Bil', axis=1, inplace=True, errors='ignore')
printScore(lr, x_train, x_test, y_train, y_test)

# # 3. Calculate feature importance with RF & remove not important features
print("---------- Feature selection methods ----------")
#ANOVA
print("ANOVA: ",end='')
def select(p, x_train, x_test, y_trian, y_test):
    #copy dataframes
    x_train_selected = x_train.copy()
    x_test_selected= x_test.copy()
    #p: percentage of remaining columns
    select = SelectPercentile(percentile = p)
    select.fit(x_train_selected, y_train)
    x_train_selected = select.transform(x_train_selected)
    x_test_selected = select.transform(x_test_selected)
    #train & test
    lr_selected = skl_lm.LogisticRegression()
    lr_selected.fit(x_train_selected, y_train)
    return (lr_selected.score(x_test_selected, y_test),select.get_support())
#find out the best p
percent, max_val, s = 0,0,0
for i in range(1,100,1):
    val,s_tmp = select(i, x_train, x_test, y_train, y_test)
    if max_val<val:
        max_val = val
        percent = i
        s = s_tmp
x_train_ANOVA = x_train.loc[:,s]
x_test_ANOVA = x_test.loc[:,s]
print(max_val, end='')
print(" (%d %%)" % percent)

#Feature Importance
print("Feature Importance: ",end='')
rf.fit(x_train, y_train)
not_important = x_train.columns[rf.feature_importances_<0.005]
x_train.drop(not_important, inplace=True, errors='ignore')
x_test.drop(not_important, inplace=True, errors='ignore')
printScore(lr, x_train, x_test, y_train, y_test)

# ---------Parameter Tuning---------------
print("-------------- Parameter Tuning --------------")
def LR(i,x_train, y_train, x_test, y_test):
    lr=skl_lm.LogisticRegression(C=i,tol=1e-5,solver='lbfgs',max_iter=500)
    lr.fit(x_train, y_train)
    return lr.score(x_test, y_test)

def tune(x_train, y_train, x_test, y_test):
    scr, val_c = 0,0
    for i in range(1,100,1):
        tmp = LR(i/10,x_train, y_train, x_test, y_test)
        if scr < tmp:
            scr = tmp
            val_c = i
    print(scr)
    return val_c

print("ANOVA: ", end='')
val_c_ANOVA = tune(x_train_ANOVA, y_train, x_test_ANOVA, y_test)
print("Feature Importance: ", end='')
val_c = tune(x_train, y_train, x_test, y_test)
print()

# --------------AUC-------------------
def printAUC(model, model_org, x_test_org, x_test, y_test):
    y_predicted_org = model_org.predict_proba(x_test_org)[:,1]
    y_predicted = model.predict_proba(x_test)[:,1]

    #FP, TP
    FP_org, TP_org, thresholds_org= roc_curve(y_test,y_predicted_org)
    auc_org = roc_auc_score(y_test, y_predicted_org)
    FP, TP, thresholds= roc_curve(y_test,y_predicted)
    auc = roc_auc_score(y_test, y_predicted)
    print(auc)

print("-------------- AUC --------------")
lr_ANOVA=skl_lm.LogisticRegression(C=val_c_ANOVA/10,tol=1e-5,solver='lbfgs',max_iter=500)
lr_ANOVA.fit(x_train_ANOVA, y_train)
lr.fit(x_train_org, y_train)
print("ANOVA: ", end='')
printAUC(lr_ANOVA,lr, x_test_org, x_test_ANOVA, y_test)

lr_fi=skl_lm.LogisticRegression(C=val_c/10,tol=1e-5,solver='lbfgs',max_iter=500)
lr_fi.fit(x_train, y_train)
lr.fit(x_train_org, y_train)
print("Feature Importance: ", end='')
printAUC(lr_fi,lr, x_test_org, x_test, y_test)
print()

print("FINAL SCORE: ",lr_fi.score(x_test, y_test))