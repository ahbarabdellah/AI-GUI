
#### --- 0. Utilities
from utilities import *

import warnings
import numpy as np 
import pandas as pd 


from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


warnings.filterwarnings('ignore')


#### --- 1. Linear Regression
def linearregression(path,features,target):
    
    # -- Data Preprocessing --
    df=pd.read_csv(path)
    X=df[features]
    Y=df[target]

    # -- Calling preprocessing functions on the feature and target set. --
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])
    X=EncodeX(X)
    Y=NullClearner(Y)

    # -- Splitting the dataset into the Training set and Test set --
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)

    # -- Fitting Linear Regression to the Training set --
    model=LinearRegression()
    model.fit(x_train,y_train)

    # -- Predicting the Test set results --
    y_pred=model.predict(x_test)

    # -- evalution metric between (y_test,y_pred) --
    accuracy = model.score(x_test,y_test)*100    
    r2 =r2_score(y_test,y_pred)*100   #r2_score
    score2 = mean_absolute_error(y_test,y_pred) #r2_score
    score3 = mean_squared_error(y_test,y_pred)

    return accuracy, r2, score2, score3, model


#### --- 2. Decision Tree Regressor
def DTRegressor(path,features,target):
    
    # -- Data Preprocessing --
    df=pd.read_csv(path)
    X=df[features]
    Y=df[target]

    # -- Calling preprocessing functions on the feature and target set. --
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])
    X=EncodeX(X)
    Y=NullClearner(Y)

    # -- Splitting the dataset into the Training set and Test set --
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)

    # -- Fitting Linear Regression to the Training set --
    model=DecisionTreeRegressor(random_state=123)
    model.fit(x_train,y_train)

    # -- Predicting the Test set results --
    y_pred=model.predict(x_test)

    # -- evalution metric between (y_test,y_pred) --
    accuracy = model.score(x_test,y_test)*100    
    r2 =r2_score(y_test,y_pred)*100   #r2_score
    score2 = mean_absolute_error(y_test,y_pred) #r2_score
    score3 = mean_squared_error(y_test,y_pred)

    return accuracy, r2, score2, score3, model


#### --- 3. GradientboostingRegressor
def GBRegressor(path,features,target):
    
    # -- Data Preprocessing --
    df=pd.read_csv(path)
    X=df[features]
    Y=df[target]

    # -- Calling preprocessing functions on the feature and target set. --
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])
    X=EncodeX(X)
    Y=NullClearner(Y)

    # -- Splitting the dataset into the Training set and Test set --
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)

    # -- Fitting Linear Regression to the Training set --
    model=GradientBoostingRegressor(random_state=123)
    model.fit(x_train,y_train)

    # -- Predicting the Test set results --
    y_pred=model.predict(x_test)

    # -- evalution metric between (y_test,y_pred) --
    accuracy = model.score(x_test,y_test)*100    
    r2 =r2_score(y_test,y_pred)*100   #r2_score
    score2 = mean_absolute_error(y_test,y_pred) #r2_score
    score3 = mean_squared_error(y_test,y_pred)

    return accuracy, r2, score2, score3, model


#### --- 4. KNeighbors regression
def KNRegressor(path,features,target):
    
    # -- Data Preprocessing --
    df=pd.read_csv(path)
    X=df[features]
    Y=df[target]

    # -- Calling preprocessing functions on the feature and target set. --
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])
    X=EncodeX(X)
    Y=NullClearner(Y)

    # -- Splitting the dataset into the Training set and Test set --
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)

    # -- Fitting Linear Regression to the Training set --
    model=KNeighborsRegressor(n_jobs=-1)
    model.fit(x_train,y_train)

    # -- Predicting the Test set results --
    y_pred=model.predict(x_test)

    # -- evalution metric between (y_test,y_pred) --
    accuracy = model.score(x_test,y_test)*100    
    r2 =r2_score(y_test,y_pred)*100   #r2_score
    score2 = mean_absolute_error(y_test,y_pred) #r2_score
    score3 = mean_squared_error(y_test,y_pred)

    return accuracy, r2, score2, score3, model


#### --- 5. LassoRegressor
def LassoRegressor(path,features,target):
    
    # -- Data Preprocessing --
    df=pd.read_csv(path)
    X=df[features]
    Y=df[target]

    # -- Calling preprocessing functions on the feature and target set. --
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])
    X=EncodeX(X)
    Y=NullClearner(Y)

    # -- Splitting the dataset into the Training set and Test set --
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)

    # -- Fitting Linear Regression to the Training set --
    model=Lasso(random_state=123)
    model.fit(x_train,y_train)

    # -- Predicting the Test set results --
    y_pred=model.predict(x_test)

    # -- evalution metric between (y_test,y_pred) --
    accuracy = model.score(x_test,y_test)*100    
    r2 =r2_score(y_test,y_pred)*100   #r2_score
    score2 = mean_absolute_error(y_test,y_pred) #r2_score
    score3 = mean_squared_error(y_test,y_pred)

    return accuracy, r2, score2, score3, model


#### --- 6. Random Forest Regressor
def RFRegressor(path,features,target):
    
    # -- Data Preprocessing --
    df=pd.read_csv(path)
    X=df[features]
    Y=df[target]

    # -- Calling preprocessing functions on the feature and target set. --
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])
    X=EncodeX(X)
    Y=NullClearner(Y)

    # -- Splitting the dataset into the Training set and Test set --
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)

    # -- Fitting Linear Regression to the Training set --
    model=RandomForestRegressor(random_state=123)
    model.fit(x_train,y_train)

    # -- Predicting the Test set results --
    y_pred=model.predict(x_test)

    # -- evalution metric between (y_test,y_pred) --
    accuracy = model.score(x_test,y_test)*100    
    r2 =r2_score(y_test,y_pred)*100   #r2_score
    score2 = mean_absolute_error(y_test,y_pred) #r2_score
    score3 = mean_squared_error(y_test,y_pred)

    return accuracy, r2, score2, score3, model






