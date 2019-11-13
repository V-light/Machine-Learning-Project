
# coding: utf-8

# In[63]:


import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import json
import pickle
import os

def dummy_row(df):
    df2 = pd.DataFrame()
    for k in df:
        df2[k] = [0]
    df2.to_csv("dummy_row.csv", index=False)
    print(df.iloc[0])

def from_dummy_row(user_df):
    df = pd.read_csv(os.path.dirname(__file__) + "/" + "dummy_row.csv")
    for k in user_df:
        df[k] =user_df[k]
    print(df.info())
    return df

def getJSON(csvFile, index=0):
    import pandas as pd
    df = pd.read_csv(csvFile)
    print(df.iloc[index].to_json())


def getDF(d1):
    d2 = {}
    for k in d1:
        print("="*70)
        print(k)
        v = []
        v.append(d1[k])
        d2[k]= v
    return pd.DataFrame(d2)


def get_title(s1):
    val = 1;
    title = s1.split(",")[1].split(".")[0].strip()
    if title == "Mr":
        val = 0
    elif title == "Mrs" or title == "Miss" or title == "Mlle" or title == "Lady":
        val = 10
    elif title == "Master":
        val = 5
    return val

def pre_proc(df):
    df["Title"] = df["Name"].apply(get_title)
    df["Cabin"] = df["Cabin"].isna()
    df.drop(["Name", "Ticket"], inplace=True, axis=1)
    df= pd.get_dummies(df)
    mAge = 30
    df.Age = df.Age.fillna(mAge)
    return df

def training():
    df = pd.read_csv("titanic.csv")
    df = pre_proc(df)
    dummy_row(df)
    y = df["Survived"]
    X = df.drop("Survived", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
    model = XGBClassifier(n_estimators=187)
    model.fit(X, y)
    with open("mymodel.pkl", "wb") as f1:
        pickle.dump(model, f1)
    print(model.score(X_test, y_test))
    y_prd = model.predict(X_test)
    print(confusion_matrix(y_test, y_prd))
    print("Actual Not Survived :: " + str(len(y_test[y_test==0])))
def pred(jsonStr):
    # x = json.loads(jsonStr)
    x = jsonStr
    df = getDF(x)
    df = pre_proc(df)
    df = from_dummy_row(df)
    model = XGBClassifier(n_estimators=187)
    if "Survived" in df:
        df.drop("Survived", axis=1, inplace=True)
    with open(os.path.dirname(__file__) + "/" + "mymodel.pkl", "rb") as f1:
        model = pickle.load(f1)
    return model.predict(df)[0]
# training()
# pred('{"PassengerId":1,"Survived":0,"Pclass":3,"Name":"Braund, Mr. Owen Harris","Sex":"male","Age":22.0,"SibSp":1,"Parch":0,"Ticket":"A\/5 21171","Fare":7.25,"Cabin":null,"Embarked":"S"}')

