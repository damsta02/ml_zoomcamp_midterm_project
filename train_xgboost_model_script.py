
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mutual_info_score
)
from sklearn.model_selection import KFold
import pickle 

import xgboost as xgb





import kagglehub

# Download latest version
path = kagglehub.dataset_download("hopesb/student-depression-dataset",
    output_dir="./student_depression_dataset")

print("Path to dataset files:", path)





df= pd.read_csv(path+"/Student Depression Dataset.csv")
df.head()


df.columns=df.columns.str.lower().str.replace(" ", "_")
for col in df.columns:
    if (df[col].dtype=="str"):
        df[col]= df[col].str.lower().str.replace(" ","_")




del df["id"]
del df["profession"]
del df["work_pressure"]
del df["job_satisfaction"]
del df["city"]


df=df.dropna().reset_index(drop=True)
categorical_columns=["gender","sleep_duration","dietary_habits","degree","have_you_ever_had_suicidal_thoughts_?","family_history_of_mental_illness"]
numerical_columns=["age","academic_pressure","cgpa","study_satisfaction","work/study_hours","financial_stress"]
df.head()


from sklearn.model_selection import train_test_split

df_full_train, df_test=train_test_split(df, test_size=0.2, random_state=5234)
df_train, df_val=train_test_split(df_full_train, test_size=0.25, random_state=5234)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train=df_train["depression"]
y_val=df_val["depression"]
y_test=df_test["depression"]


del df_train["depression"]
del df_val["depression"]
del df_test["depression"]


dv=DictVectorizer(sparse=False)

X_train=dv.fit_transform(df_train.to_dict(orient="records"))
X_val=dv.transform(df_val.to_dict(orient="records"))
X_test=dv.transform(df_test.to_dict(orient="records"))
features = list(dv.get_feature_names_out())
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)

dtrain=xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval=xgb.DMatrix(X_val, label=y_val, feature_names=features)
dtest=xgb.DMatrix(X_test, label=y_test, feature_names=features)




xgb_params = {
    'eta': 0.05,
    'max_depth': 10,
    'min_child_weight': 5,

    'objective': 'binary:logistic',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200)


y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)


def train_xgboost(params,dtrain, num_rounds):
    return xgb.train(params, dtrain, num_boost_round=num_rounds)


eta_list=[0.05, 0.09 ,0.1, 0.2, 0.3]

for eta in eta_list:
    xgb_params = {
        'eta': eta,
        'max_depth': 10,
        'min_child_weight': 5,

        'objective': 'binary:logistic',
        'nthread': 8,

        'seed': 1,
        'verbosity': 1,
    }
    model = train_xgboost(xgb_params, dtrain, 200)
    y_pred = model.predict(dval)
    print(f"{eta}:",roc_auc_score(y_val, y_pred))



max_depth_list=[2,3,4,5,7,10,20, 30]

for md in max_depth_list:
    xgb_params = {
        'eta': 0.05,
        'max_depth': md,
        'min_child_weight': 5,

        'objective': 'binary:logistic',
        'nthread': 8,

        'seed': 1,
        'verbosity': 1,
    }
    model = train_xgboost(xgb_params, dtrain, 200)
    y_pred = model.predict(dval)
    print(f"{md}:",roc_auc_score(y_val, y_pred))



min_child_weight_list=[2,3,4,5,7,10,20, 30]

for mcw in min_child_weight_list:
    xgb_params = {
        'eta': 0.05,
        'max_depth': 3,
        'min_child_weight': mcw,

        'objective': 'binary:logistic',
        'nthread': 8,

        'seed': 1,
        'verbosity': 1,
    }
    model = train_xgboost(xgb_params, dtrain, 200)
    y_pred = model.predict(dval)
    print(f"{mcw}:",roc_auc_score(y_val, y_pred))



num_rounds=[20,40,70,120,160,200,250,300,400]

for nr in num_rounds:
    xgb_params = {
        'eta': 0.05,
        'max_depth': 3,
        'min_child_weight': 4,

        'objective': 'binary:logistic',
        'nthread': 8,

        'seed': 1,
        'verbosity': 1,
    }
    model = train_xgboost(xgb_params, dtrain, num_rounds=nr)
    y_pred = model.predict(dval)
    print(f"{nr}:",roc_auc_score(y_val, y_pred))



kfold= KFold(n_splits=5,shuffle=True, random_state=2344)
xgb_params = {
        'eta': 0.05,
        'max_depth': 3,
        'min_child_weight': 4,

        'objective': 'binary:logistic',
        'nthread': 8,

        'seed': 1,
        'verbosity': 1,
    }

scores=[]

for train_idx, val_idx in kfold.split(df_full_train):
    df_train=df_full_train.iloc[train_idx]
    df_val=df_full_train.iloc[val_idx]
    y_train = df_train.depression.values
    y_val = df_val.depression.values
    df_train=df_train.drop(columns=["depression"])
    df_val=df_val.drop(columns=["depression"])
    X_train=dv.transform(df_train.to_dict(orient="records"))
    X_val=dv.transform(df_val.to_dict(orient="records"))
    dtrain=xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval=xgb.DMatrix(X_val, label=y_val, feature_names=features)
    model=train_xgboost(dtrain=dtrain, params=xgb_params,num_rounds=250 )
    y_pred=model.predict(dval)
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)




scores


xgb_params = {
        'eta': 0.05,
        'max_depth': 3,
        'min_child_weight': 4,

        'objective': 'binary:logistic',
        'nthread': 8,

        'seed': 1,
        'verbosity': 1,
    }

X_train=dv.transform(df_full_train.to_dict(orient="records"))
y_train=df_full_train["depression"]
dtrain=xgb.DMatrix(X_train, label=y_train, feature_names=features)
final_model=train_xgboost(dtrain=dtrain, params=xgb_params,num_rounds=250)
y_pred=model.predict(dtest)
auc = roc_auc_score(y_test, y_pred)





y_pred_class = (y_pred >= 0.5).astype(int)


accuracy = accuracy_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
auc = roc_auc_score(y_test, y_pred)
precision = precision_score(y_test, y_pred_class)


print("Accuracy:  ", round(accuracy, 4))
print("Recall:    ", round(recall, 4))
print("F1 Score:  ", round(f1, 4))
print("AUC:       ", round(auc, 4))
print("Precision: ", round(precision, 4))


#with open("xg_boost_model.bin", "wb") as fout:
#    pickle.dump((dv, final_model), fout)




