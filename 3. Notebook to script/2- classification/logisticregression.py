from utilities import *

import warnings 
import numpy as np 
import pandas as pd 
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')



def logisticregression(path,features,target):
    
    # -- Data Preprocessing --
    df=pd.read_csv(path)
    X = df[features]
    Y = df[target]

    # -- Calling preprocessing functions on the feature and target set. --
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])  
    X=EncodeX(X)
    Y=EncodeY(NullClearner(Y))
    
    # -- Splitting the dataset into the Training set and Test set --
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    
    # Build Model here
    model = LogisticRegression(random_state = 123,n_jobs = -1)
    model.fit(x_train, y_train)
    return model,x_train,x_test,y_train,y_test
