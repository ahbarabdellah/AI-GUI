# -- Importing the libraries --
from utilities import *

import warnings
import numpy as np 
import pandas as pd 
import seaborn as se 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')




def linar_reg(path,features,target):
    
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
    score1 = model.score(x_test,y_test)*100    #r2_score
    score2 = mean_absolute_error(y_test,y_pred) #r2_score
    score3 = mean_squared_error(y_test,y_pred)
    score4 = mean_absolute_error(y_test,y_pred)
    score5 = np.sqrt(mean_squared_error(y_test,y_pred))

    return score1, score2, score3, score4, score5, model
