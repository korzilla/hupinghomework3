# coding: utf-8
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def get_data():
    data_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(data_file)
    return df


df = get_data()
df.info()
def preprocess_data():
    global df

    # 数据清洗,删除无用特征，axis=1是按照列操作。
    df.drop(['customerID'], axis=1, inplace=True)
    df.drop(['Contract'], axis=1, inplace=True)
    df.drop(['PaymentMethod'], axis=1, inplace=True)
    
    # 'TotalCharges' 列被识别为object，将其强制转换为float64失败，后发现其中有几行数据的值是空格，因此将这几行数据删除
    tc = df['TotalCharges']
    """
    for i in tc:
        try:
            a = float(i)
        except:
            print(f"=={i}==")
    """
    todel = []
    for i in range(len(tc)):
        if tc[i] == ' ':
            #print(f"line {i} null value")
            todel.append(i)
    #df = df.drop(todel)
    df.drop(todel, inplace=True)
    df['TotalCharges'] = df['TotalCharges'].astype('float64') 
    print(df.shape)

    gender_mapper = {'Male': 1, 'Female': 0}
    df['gender'].replace(gender_mapper, inplace=True)

    yesno_mapper = {'Yes':1, 'No':0, 'No phone service': 2, 'No internet service': 2}
    inets_mapper = {'No':0, 'DSL':1, 'Fiber optic':2}
    df['Partner'].replace(yesno_mapper, inplace=True)
    df['Dependents'].replace(yesno_mapper, inplace=True)
    df['PhoneService'].replace(yesno_mapper, inplace=True)
    df['MultipleLines'].replace(yesno_mapper, inplace=True)

    df['InternetService'].replace(inets_mapper, inplace=True)
    df['OnlineSecurity'].replace(yesno_mapper, inplace=True)
    df['OnlineBackup'].replace(yesno_mapper, inplace=True)
    df['DeviceProtection'].replace(yesno_mapper, inplace=True)
    df['TechSupport'].replace(yesno_mapper, inplace=True)
    df['StreamingTV'].replace(yesno_mapper, inplace=True)
    df['StreamingMovies'].replace(yesno_mapper, inplace=True)
    df['PaperlessBilling'].replace(yesno_mapper, inplace=True)
    df['Churn'].replace(yesno_mapper, inplace=True)

preprocess_data()
df.info()
print("===============")
print(df['TotalCharges'].values)
#print(df['gender'].values)
#print(df['Partner'].values)
print(df.values)

def select_signature():
    X = df.drop('Churn', axis=1)
    Y = df['Churn']
    r = RFE(estimator=LogisticRegression(), n_features_to_select=5).fit_transform(X, Y)
    
    print(r)
    return r
print("====process data====")
sdata = select_signature()

x = sdata
y = df['Churn']
xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.2)
#for i in [xTrain,xTest,yTrain,yTest]:
#    i = i.reset_index()

clf = tree.DecisionTreeClassifier(random_state=0
                             ,max_depth=10
                             ,criterion="entropy"
                             )
clf = clf.fit(xTrain,yTrain)


y_pred_default = clf.predict(xTest)
print(classification_report(yTest, y_pred_default))
#print(accuracy_score(yTest,y_pred_default))
"""
scoreTest = []
scoreTrain = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(random_state=0
                                 ,max_depth=i+1
                                 ,criterion="entropy"
                                 )
    clf = clf.fit(xTrain,yTrain)
    onceTrain = clf.score(xTrain,yTrain)
    #onceTest = tree.cross_val_score(clf,x,y,cv=10).mean()
    onceTest = clf.score(xTest,yTest)
    scoreTest.append(onceTest)
    scoreTrain.append(onceTrain)
print(max(scoreTest))
plt.figure()
plt.plot(range(1,11),scoreTrain,color="red",label="train")
plt.plot(range(1,11),scoreTest,color="blue",label="test")
plt.xticks(range(1,11))
plt.legend()
plt.show()
"""
