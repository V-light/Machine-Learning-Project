#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os


def dummy_row(df):
    df2 = pd.DataFrame()
    for k in df:
        df2[k] = [0]
    df2.to_csv("dummy_row.csv", index=False)
    
def from_dummy_row(user_df):
    df = pd.read_csv(os.path.dirname(__file__)+ "/" +"dummy_row.csv")
    for k in user_df:
        df[k] = user_df[k]
    return df

def getDF(d1):
    d2 ={}
    for k in d1:
        v = []
        v.append(d1[k])
        d2[k] = v
    return pd.DataFrame(d2)


        
def pre_proc(df):
    df.salary = df.salary.replace(["low", "medium", "high"], [0,1,2])
    df.salary = df.salary.astype(int)
    df = pd.get_dummies(df)
    return df

def training():
    df = pd.read_csv("HR_comma_sep.csv")
    df   = pre_proc(df)
    dummy_row(df)

    y = df["left"]
    X = df.drop("left", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
    model = RandomForestClassifier(n_estimators=300, random_state=1)
    model.fit(X_train, y_train)
    with open("mymodel.pkl", "wb") as f1:
        pickle.dump(model, f1)
    print(model.score(X_test, y_test))
    from sklearn.metrics import confusion_matrix
    y_prd = model.predict(X_test)
    return confusion_matrix(y_test, y_prd)


def pred(jsonStr):
    x =json.loads(jsonStr)
    df =getDF(x)
    df = pre_proc(df)
    df = from_dummy_row(df)
    if "left" in df:
        df.drop("left", axis=1, inplace=True)  
        
    with open( os.path.dirname(__file__)+ "/"+"mymodel.pkl", 'rb') as f1:
        model = pickle.load(f1)
        
    return model.predict(df)[0]
#     return df
# training()
# pred('{"satisfaction_level":0.38,"last_evaluation":0.53,"number_project":2,"average_montly_hours":157,"time_spend_company":3,"Work_accident":0,"left":1,"promotion_last_5years":0,"Department":"sales","salary":"low"}')


# In[7]:


df = pd.read_csv("HR_comma_sep.csv")


# In[11]:


df.iloc[0].to_json()


# In[ ]:




