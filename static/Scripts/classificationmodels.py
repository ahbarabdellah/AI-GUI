### --- 0. the utilities.py conatin the function preproccessAndSplit() and scores()
from static.Scripts.utilities  import *
import warnings

from imblearn.over_sampling import RandomOverSampler # need to be installed !pip install imblearn
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier 
warnings.filterwarnings('ignore')



#### ----  1. Logistique Regression :
def logisticregression(df,features,target):
    X, x_train,x_test,y_train,y_test = preproccessAndSplit(df,features,target)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model = LogisticRegression(random_state = 123,n_jobs = -1)
    model.fit(x_train, y_train)
    return X, x_test, y_test, model

#### ----  2. KNN (KNeighborsClassifier) :
def KNClassifier(df,features ,target   ):
    X, x_train,x_test,y_train,y_test = preproccessAndSplit(df,features,target)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model=KNeighborsClassifier(n_jobs=-1)
    model.fit(x_train,y_train)
    return X, x_test, y_test, model

#### ----  3. Decision Tree :
def DTClassifier(df,features,target):
    X, x_train,x_test,y_train,y_test = preproccessAndSplit(df,features,target)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model = DecisionTreeClassifier(random_state=123)
    model.fit(x_train,y_train)
    return X, x_test, y_test, model

#### ----  4. Perceptron :
def perceptron(df,features,target):
    X, x_train,x_test,y_train,y_test = preproccessAndSplit(df,features,target)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model = Perceptron(random_state=123)
    model.fit(x_train, y_train)
    return X, x_test, y_test, model

#### ----  5. SVC :
def scv(df,features,target):
    X, x_train,x_test,y_train,y_test = preproccessAndSplit(df,features,target)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model = SVC(random_state = 123)
    model.fit(x_train,y_train)
    return X, x_test, y_test, model

#### ----  6. Linear SVC:
def LinearSvc(df,features,target):
    X, x_train,x_test,y_train,y_test = preproccessAndSplit(df,features,target)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model=LinearSVC(random_state=123)
    model.fit(x_train,y_train)
    return X, x_test, y_test, model

#### ----  7. Random forest Classifier :
def RFClassifier(df , features, target):
    X, x_train,x_test,y_train,y_test = preproccessAndSplit(df,features,target)
    model = RandomForestClassifier(n_jobs = -1,random_state = 123)
    model.fit(x_train, y_train)
    return X, x_test, y_test, model