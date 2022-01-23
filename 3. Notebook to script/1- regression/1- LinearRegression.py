'''
To DO list : 
1. Correlation matrix (heat map) to image to use in tkinter (GUI)
2. transform the the plots of prediction to imgs 
3. return them to the GUI
'''
# -- Importing the libraries --

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


### Data Preprocessing

'''
Since the majority of the machine learning models in the Sklearn library doesn't
handle string category data and Null value, we have to explicitly remove or replace
null values. The below snippet have functions, which removes the null value if any
exists. And convert the string classes data in the datasets by encoding them to
integer classes.

'''

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


def linar_reg(dataset,features,target):
    # Linar regression model :

    # -- Importing the dataset --
    #filepath
    file_path= dataset #'dataset.csv'


    #List of features which are  required for model training .
    features=features # array of features

    target=   target  #y_value

    # -- Data Preprocessing --
    df=pd.read_csv(file_path)
    X=df[features]
    Y=df[target]

    # -- Calling preprocessing functions on the feature and target set. --
    x=X.columns.to_list()
    for i in x:
        X[i]=NullClearner(X[i])
    X=EncodeX(X)
    Y=NullClearner(Y)
    X.head()

    f,ax = plt.subplots(figsize=(18, 18))
    matrix = np.triu(X.corr())
    se.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, mask=matrix)
    plt.show()

    # -- Splitting the dataset into the Training set and Test set --
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)

    # -- Fitting Linear Regression to the Training set --
    model=LinearRegression()
    model.fit(x_train,y_train)

    # -- Predicting the Test set results --
    y_pred=model.predict(x_test)

    #evalution metric between (y_test,y_pred)
    score1 = model.score(x_test,y_test)*100    #r2_score
    score2 = mean_absolute_error(y_test,y_pred) #r2_score
    score3 = mean_squared_error(y_test,y_pred)
    score4 = mean_absolute_error(y_test,y_pred)
    score5 = np.sqrt(mean_squared_error(y_test,y_pred))


    # -- Visualising the Training set results --
    plt.figure(figsize=(14,10))
    plt.plot(range(20),y_test[0:20], color = "green")
    plt.plot(range(20),model.predict(x_test[0:20]), color = "red")
    plt.legend(["Actual","prediction"]) 
    plt.title("Predicted vs True Value")
    plt.xlabel("Record number")
    plt.ylabel(target)
    plt.show()
