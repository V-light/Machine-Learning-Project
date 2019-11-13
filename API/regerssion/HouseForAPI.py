#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import os



def dummy_row(df):
    df2 = pd.DataFrame()
    for i in df:
        df2[i] = [0]
        
    df2.to_csv("dummy_row.csv", index = False)
    
def from_dummy_row(u_df):
    df = pd.read_csv(os.path.dirname(__file__)+ "/"+ "dummy_row.csv")
    for i in u_df:
        df[i] = u_df[i]
        
    return df
    
def getDf(df):
    d2 = {}
    for i in df:
        v = []
        v.append(df[i])
        d2[i] = v
    return pd.DataFrame(d2)
    
def  fun(x):
    return int(x.split("T")[0])



def pre_pro(df):
    
    df["date"] = df["date"].apply(fun)
    df.drop(["date"], axis = 1, inplace = True)
    return df
    
    
def training():
    df  = pd.read_csv("kc_house_data.csv")
    df = pre_pro(df)
    dummy_row(df)
    
    y= df["price"]
    x = df.drop(['price'], axis = 1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size= 0.20, random_state = 11)

    model = XGBRegressor(n_estimators=1500, random_state= 11)
    model.fit(x, y)
    with open("my_model.pkl", "wb") as f1:
        pickle.dump(model, f1)
    
    
def pred(jsonStr):
    x = json.loads(jsonStr)
    
    df = getDf(x)
    df = pre_pro(df)
    df = from_dummy_row(df)
    
    if "price" in df:
        df.drop("price", axis =1 , inplace = True)
        
    with open(os.path.dirname(__file__)+ "/"+ "my_model.pkl" , "rb") as f1:
        model = pickle.load(f1)
    return model.predict(df)[0]
    
# training()
# pred('{"id":5631500400,"date":"20150225T000000","price":180000.0,"bedrooms":2,"bathrooms":1.0,"sqft_living":770,"sqft_lot":10000,"floors":1.0,"waterfront":0,"view":0,"condition":3,"grade":6,"sqft_above":770,"sqft_basement":0,"yr_built":1933,"yr_renovated":0,"zipcode":98028,"lat":47.7379,"long":-122.233,"sqft_living15":2720,"sqft_lot15":8062}')

# pred('{"id":2487200875,"date":"20141209T000000","price":604000.0,"bedrooms":4,"bathrooms":3.0,"sqft_living":1960,"sqft_lot":5000,"floors":1.0,"waterfront":0,"view":0,"condition":5,"grade":7,"sqft_above":1050,"sqft_basement":910,"yr_built":1965,"yr_renovated":0,"zipcode":98136,"lat":47.5208,"long":-122.393,"sqft_living15":1360,"sqft_lot15":5000}')

