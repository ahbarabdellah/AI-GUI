### --- 0. the utilities.py conatin the function preproccessAndSplit() and scores()
from utilities import *
import warnings

from sklearn.linear_model import LinearRegression,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
warnings.filterwarnings('ignore')


#### --- 1. Linear Regression
def linearregression(df,features,target):
    
    x_train,x_test,y_train,y_test = preproccessAndSplit(df,features,target)
    model=LinearRegression()
    model.fit(x_train,y_train)
    return scores(x_test,y_test,model ), model

#### --- 2. Decision Tree Regressor
def DTRegressor(df,features,target):
    
    x_train,x_test,y_train,y_test = preproccessAndSplit(df,features,target)
    model=DecisionTreeRegressor(random_state=123)
    model.fit(x_train,y_train)
    return scores(x_test,y_test,model ), model

#### --- 3. GradientboostingRegressor
def GBRegressor(df,features,target):
    
    x_train,x_test,y_train,y_test = preproccessAndSplit(df,features,target)
    model=GradientBoostingRegressor(random_state=123)
    model.fit(x_train,y_train)
    return scores(x_test,y_test,model ), model

#### --- 4. KNeighbors regression
def KNRegressor(df,features,target):
    
    x_train,x_test,y_train,y_test = preproccessAndSplit(df,features,target)
    model=KNeighborsRegressor(n_jobs=-1)
    model.fit(x_train,y_train)
    return scores(x_test,y_test,model ), model

#### --- 5. LassoRegressor
def LassoRegressor(df,features,target):
    
    x_train,x_test,y_train,y_test = preproccessAndSplit(df,features,target)
    model=Lasso(random_state=123)
    model.fit(x_train,y_train)
    return scores(x_test,y_test,model ), model

#### --- 6. Random Forest Regressor
def RFRegressor(df,features,target):
    
    x_train,x_test,y_train,y_test = preproccessAndSplit(df,features,target)
    model=RandomForestRegressor(random_state=123)
    model.fit(x_train,y_train)
    return scores(x_test,y_test,model ), model
