import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 


def NullClearner(df):
    if(isinstance(df, pd.Series) and (df.dtype in ["float64","int64"])):
        df.fillna(df.mean(),inplace=True)
        return df
    elif(isinstance(df, pd.Series)):
        df.fillna(df.mode()[0],inplace=True)
        return df
    else:return df
    
    
def EncodeX(df):
    return pd.get_dummies(df)
def EncodeY(df):
    if len(df.unique())<=2:
        return df
    else:
        un_EncodedT=np.sort(pd.unique(df), axis=-1, kind='mergesort')
        df=LabelEncoder().fit_transform(df)
        EncodedT=[xi for xi in range(len(un_EncodedT))]
        return df


#### --- Pre-defined functions (will be used in all algos)

def preproccessAndSplit(path,features,target):
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
    return x_train,x_test,y_train,y_test
    

def scores(x_test,y_test,model ):
    y_pred=model.predict(x_test)
    accuracy = model.score(x_test,y_test)*100    
    r2 =r2_score(y_test,y_pred)*100   #r2_score
    score2 = mean_absolute_error(y_test,y_pred) #r2_score
    score3 = mean_squared_error(y_test,y_pred)
    return accuracy, r2, score2, score3