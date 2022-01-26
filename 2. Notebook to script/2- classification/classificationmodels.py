
#### ----  0. Utelities : 
# !pip install imblearn
from utilities import *
import numpy as np 
import pandas as pd 

import warnings

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier 

from sklearn.svm import SVC
warnings.filterwarnings('ignore')



#### ----  1. Logistique Regression :
def logisticregression(path,features,target):
    df=pd.read_csv(path)
    #df.index= df.columns.to_list()
    X = df[features]
    Y = df[target]
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])  
    X=EncodeX(X)
    Y=EncodeY(NullClearner(Y))
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
        # Build Model here
    model = LogisticRegression(random_state = 123,n_jobs = -1)
    model.fit(x_train, y_train)
    return model,x_train,x_test,y_train,y_test

#### ----  2. KNN (KNeighborsClassifier) :
def KNN(file_path,features ,target   ):
    df=pd.read_csv(file_path)
    X=df[features]
    Y=df[target]
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])
    X=EncodeX(X)
    Y=EncodeY(NullClearner(Y))
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model=KNeighborsClassifier(n_jobs=-1)
    model.fit(x_train,y_train)
    return model,x_train,x_test,y_train,y_test

#### ----  3. Decision Tree :
def DecisionTree(file_path,features,target):
    df=pd.read_csv(file_path)
    X=df[features]
    Y=df[target]
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])  
    X=EncodeX(X)
    Y=EncodeY(NullClearner(Y))
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model = DecisionTreeClassifier(random_state=123)
    model.fit(x_train,y_train)
    return model,x_train,x_test,y_train,y_test

#### ----  4. Perceptron :
def perseptron(path,features,target):
    df=pd.read_csv(path)
    X = df[features]
    Y = df[target]
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])  
    X=EncodeX(X)
    Y=EncodeY(NullClearner(Y))
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model = Perceptron(random_state=123)
    model.fit(x_train, y_train)
    return model,x_train,x_test,y_train,y_test

#### ----  5. SVC :
def scv(path,features,target):
    df = pd.read_csv(path)
    X = df[features]
    Y = df[target]
    x = X.columns.to_list()
    for i in x :
        X[i] = NullClearner(X[i])
    X = EncodeX(X)
    Y = EncodeY(NullClearner(Y))
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=123)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train,y_train)
    model = SVC(random_state = 123)
    model.fit(x_train,y_train)
    return model, x_train,x_test,y_train,y_test

#### ----  6. Liniar SVC:
def LiniarSvc(path,features,target):
    df = pd.read_csv(path)
    X = df[features]
    Y = df[target]
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])
    X=EncodeX(X)
    Y=EncodeY(NullClearner(Y))
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model=LinearSVC(random_state=123)
    model.fit(x_train,y_train)
    return model,x_train,x_test,y_train,y_test

#### ----  7. Random forest Classifier :
def rfc(path , features, target):
    df = pd.read_csv(path)
    X = df[features]
    Y = df[target]
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])
    X=EncodeX(X)
    Y=EncodeY(NullClearner(Y))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 123)#performing datasplitting
    # Build Model here
    model = RandomForestClassifier(n_jobs = -1,random_state = 123)
    model.fit(x_train, y_train)
    return model,x_train,x_test,y_train,y_test

